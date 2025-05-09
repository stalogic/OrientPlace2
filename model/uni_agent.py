import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from retry import retry
from loguru import logger
from pathlib import Path
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from util import trackit

from .models import UniActor, UniCritic


class UniPPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10

    def __init__(
        self,
        placed_num_macro: int,
        grid: int,
        batch_size: int,
        lr: float, #| dict[str, float],
        device: str = "cpu",
        ddp: bool = False
    ):
        super(UniPPO, self).__init__()
        self.placed_num_macro = placed_num_macro
        self.grid = grid
        self.batch_size = batch_size
        self.device = device
        self.uni_actor_net = UniActor().float().to(device)
        self.uni_critic_net = UniCritic().float().to(device)
        self.ddp = ddp
        if self.ddp:
            self.uni_actor_net = DDP(self.uni_actor_net)
            self.uni_critic_net = DDP(self.uni_critic_net)

        self.training_step = 0
        if isinstance(lr, float):
            uni_actor_lr = lr
            uni_critic_lr = lr
        elif isinstance(lr, dict):
            uni_actor_lr = lr.get("uni_actor", 1e-5)
            uni_critic_lr = lr.get("uni_critic", 1e-4)
        else:
            raise ValueError("lr must be float or dict")
        logger.info(f"{uni_actor_lr=}, {uni_critic_lr=}")
        self.uni_actor_optimizer = optim.Adam(self.uni_actor_net.parameters(), lr=uni_actor_lr)
        self.uni_critic_optimizer = optim.Adam(self.uni_critic_net.parameters(), lr=uni_critic_lr)

        self.CANVAS_SLICE = None
        self.WIRE_SLICE = None
        self.POS_SLICE = None
        self.FEATURE_SLICE = None

    @retry(tries=3, delay=1, backoff=2)
    def load_model(self, path: Path):
        with gzip.open(path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device(self.device))
            self.uni_actor_net.load_state_dict(checkpoint["uni_actor_net"])
            self.uni_critic_net.load_state_dict(checkpoint["uni_critic_net"])

    @retry(tries=3, delay=1, backoff=2)
    def save_model(self, save_path: Path, save_flag: str):
        # 根据是否使用DDP，选择保存模型的方式
        uni_actor_net = self.uni_actor_net.module if self.ddp else self.uni_actor_net
        uni_critic_net = self.uni_critic_net.module if self.ddp else self.uni_critic_net
        save_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(save_path / f"{save_flag}_state_dict.pkl.gz", "wb") as f:
            torch.save({
                "uni_actor_net": uni_actor_net.state_dict(),
                "uni_critic_net": uni_critic_net.state_dict(),
            }, f)

    @trackit
    def select_action(self, state):

        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        canvas = state[:, self.CANVAS_SLICE].reshape(-1, 1, self.grid, self.grid)
        wire_img = state[:, self.WIRE_SLICE].reshape(-1, 8, self.grid, self.grid)
        pos_mask = state[:, self.POS_SLICE].reshape(-1, 8, self.grid, self.grid)
        macro_id = state[:, -3]

        batch_size = state.shape[0]

        state = torch.where(pos_mask == 1, -wire_img, wire_img)
        state = torch.concat([canvas, state], dim=1)

        with torch.no_grad():
            self.uni_actor_net.eval()
            self.uni_critic_net.eval()

            action_prob = self.uni_actor_net(state, pos_mask)
            assert action_prob.shape == (batch_size, 8*224*224)
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            assert action_log_prob.shape == (batch_size,)

            value = self.uni_critic_net(state, macro_id)
            assert value.shape == (batch_size, 1)

        action_info = (action.item(), action_log_prob.item(), value.item())
        state_info = (state[0].cpu().numpy(), pos_mask[0].cpu().numpy(), macro_id.long().item())

        return action_info, state_info
  
    @trackit
    def update(self, data:dict=None) -> None:

        
        # 分布式训练，数据来自reverb
        macro_id = torch.tensor(data['macro_id'], dtype=torch.int64).to(self.device)
        state = torch.tensor(data['state'], dtype=torch.float).to(self.device)
        mask = torch.tensor(data['mask'], dtype=torch.float).to(self.device)

        action = torch.tensor(data['action'], dtype=torch.int64).to(self.device)
        old_log_prob = torch.tensor(data['log_prob'], dtype=torch.float).to(self.device)
        advantage = torch.tensor(data['advantage'], dtype=torch.float).to(self.device)
        target_value = torch.tensor(data['return'], dtype=torch.float).to(self.device)
        

        advantage_mean = advantage.mean().cpu().item()
        advantage_std = advantage.std().cpu().item()
        advantage_pos_rate = (advantage > 0).float().mean().cpu().item()
        advantage_neg_rate = (advantage < 0).float().mean().cpu().item()
        logger.info(f"Advantage mean: {advantage_mean:.4f}, std: {advantage_std:.4f}, pos rate: {advantage_pos_rate*100:.2f}%, neg rate: {advantage_neg_rate*100:.2f}%")

        for epoch in range(self.ppo_epoch):  # iteration ppo_epoch
            epoch_progess = f" Epoch {epoch+1} / {self.ppo_epoch} "
            logger.info(f"{epoch_progess:-^80}")

            ratio_list, actor_losses, critic_losses = [], [], []
            for index in BatchSampler(SubsetRandomSampler(range(action.shape[0])), self.batch_size, True):
                self.training_step += 1

                batch_macro_id = macro_id[index].to(self.device)
                batch_state = state[index].to(self.device)
                batch_mask = mask[index].to(self.device)

                state_dict = {
                    'macro_id': batch_macro_id,
                    'state': batch_state,
                    'mask': batch_mask,
                }

                batch_action = action[index].to(self.device)
                batch_old_log_prob = old_log_prob[index].to(self.device)
                batch_advantage = advantage[index].to(self.device)
                batch_target_value = target_value[index].to(self.device)

                ratios, actor_loss, critic_loss = self.update_agent(state_dict, batch_action, batch_target_value, batch_advantage, batch_old_log_prob)
                ratio_list.append(ratios)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)


            ratios = np.concatenate(ratio_list)
            clip_rate = np.mean(np.abs(ratios - 1) > self.clip_param)
            up_rate = np.mean(ratios > 1)
            down_rate = np.mean(ratios < 1)
            logger.info(f"actor_loss: {np.mean(actor_losses):.4e}, critic_loss: {np.mean(critic_losses):.4e}")
            logger.info(f"Ratio# clip_rate: {clip_rate*100:.2f}%, up_rate: {up_rate*100:.2f}%, down_rate: {down_rate*100:.2f}%, max: {np.max(ratios):.5f}, min: {np.min(ratios):.5f}, mean: {np.mean(ratios):.5f}, std: {np.std(ratios):.5f}")

    def update_agent(self, state_dict, batch_action, batch_target_value, batch_advantage, batch_old_log_prob):
        state = state_dict['state']
        mask = state_dict['mask']
        macro_id = state_dict['macro_id'].squeeze()
        batch_action = batch_action.squeeze()
        batch_advantage = batch_advantage.squeeze()
        batch_old_log_prob = batch_old_log_prob.squeeze()

        self.uni_actor_net.train()
        self.uni_critic_net.train()

        normalize_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

        action_probs = self.uni_actor_net(state, mask)
        log_prob = Categorical(action_probs).log_prob(batch_action)
        assert log_prob.shape == batch_old_log_prob.shape
        ratio = torch.exp(log_prob - batch_old_log_prob) # (batch_size,)

        assert ratio.shape == normalize_advantage.shape
        L1 = ratio * normalize_advantage
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * normalize_advantage
        assert L1.shape == L2.shape
        actor_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
        assert actor_loss.shape == (), f"{actor_loss.shape=}"

        self.uni_actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.uni_actor_net.parameters(), self.max_grad_norm)
        self.uni_actor_optimizer.step()

        value = self.uni_critic_net(state, macro_id)
        critic_loss = F.smooth_l1_loss(value, batch_target_value)
        self.uni_critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.uni_critic_net.parameters(), self.max_grad_norm)
        self.uni_critic_optimizer.step()

        return ratio.detach().cpu().numpy(), actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
