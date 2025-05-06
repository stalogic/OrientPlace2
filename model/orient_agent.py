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

from .models import Actor, Critic, OrientActor, OrientCritic


class OrientPPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10

    def __init__(
        self,
        placed_num_macro: int,
        grid: int,
        num_game_per_update:int,
        batch_size: int,
        lr: float, #| dict[str, float],
        gamma: float,
        device: str = "cpu",
        ddp: bool = False
    ):
        super(OrientPPO, self).__init__()
        self.placed_num_macro = placed_num_macro
        self.buffer_capacity = num_game_per_update * placed_num_macro
        self.grid = grid
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.place_actor_net = Actor().float().to(device)
        self.place_critic_net = Critic().float().to(device)
        self.orient_actor_net = OrientActor().float().to(device)
        self.orient_critic_net = OrientCritic().float().to(device)
        if ddp:
            self.place_actor_net = DDP(self.place_actor_net, device_ids=[device])
            self.place_critic_net = DDP(self.place_critic_net, device_ids=[device])
            self.orient_actor_net = DDP(self.orient_actor_net, device_ids=[device])
            self.orient_critic_net = DDP(self.orient_critic_net, device_ids=[device])

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        if isinstance(lr, float):
            place_actor_lr = lr
            place_critic_lr = lr
            orient_actor_lr = lr
            orient_critic_lr = lr
        elif isinstance(lr, dict):
            place_actor_lr = lr.get("place_actor", 1e-5)
            place_critic_lr = lr.get("place_critic", place_actor_lr)
            orient_actor_lr = lr.get("orient_actor", 1e-5)
            orient_critic_lr = lr.get("orient_critic", orient_actor_lr)
        else:
            raise ValueError("lr must be float or dict")
        logger.info(f"{place_actor_lr=}, {place_critic_lr=}, {orient_actor_lr=}, {orient_critic_lr=}")
        self.place_actor_optimizer = optim.Adam(self.place_actor_net.parameters(), place_actor_lr)
        self.place_critic_optimizer = optim.Adam(self.place_critic_net.parameters(), place_critic_lr)
        self.orient_actor_optimizer = optim.Adam(self.orient_actor_net.parameters(), orient_actor_lr)
        self.orient_critic_optimizer = optim.Adam(self.orient_critic_net.parameters(), orient_critic_lr)

        self.place_actor_scheduler = optim.lr_scheduler.StepLR(self.place_actor_optimizer, step_size=50*self.ppo_epoch, gamma=0.3)
        self.place_critic_scheduler = optim.lr_scheduler.StepLR(self.place_critic_optimizer, step_size=50*self.ppo_epoch, gamma=0.3)
        self.orient_actor_scheduler = optim.lr_scheduler.StepLR(self.orient_actor_optimizer, step_size=50*self.ppo_epoch, gamma=0.3)
        self.orient_critic_scheduler = optim.lr_scheduler.StepLR(self.orient_critic_optimizer, step_size=50*self.ppo_epoch, gamma=0.3)

        self.placer_ok = False
        self.train_orient_agent = False
        self.train_place_agent = True
        self.orient_agent_update_count = None
        self.place_agent_update_count = None

        self.CANVAS_SLICE = None
        self.WIRE_SLICE = None
        self.POS_SLICE = None
        self.FEATURE_SLICE = None

    @retry(tries=3, delay=1, backoff=2)
    def load_model(self, path: Path):
        with gzip.open(path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device(self.device))
            self.place_actor_net.load_state_dict(checkpoint["place_actor_net"])
            self.place_critic_net.load_state_dict(checkpoint["place_critic_net"])
            self.orient_actor_net.load_state_dict(checkpoint["orient_actor_net"])

    @retry(tries=3, delay=1, backoff=2)
    def save_model(self, save_path: Path, save_flag: str):
        save_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(save_path / f"{save_flag}_state_dict.pkl.gz", "wb") as f:
            torch.save(
                {
                    "place_actor_net": self.place_actor_net.state_dict(),
                    "place_critic_net": self.place_critic_net.state_dict(),
                    "orient_actor_net": self.orient_actor_net.state_dict(),
                    "orient_critic_net": self.orient_critic_net.state_dict(),
                },
                f,
            )

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    @trackit
    def select_action(self, state) -> tuple[int, int, float, float]:

        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        canvas = state[:, self.CANVAS_SLICE].reshape(-1, 1, self.grid, self.grid)
        wire_img_8oc = state[:, self.WIRE_SLICE].reshape(-1, 8, self.grid, self.grid)
        pos_mask_8oc = state[:, self.POS_SLICE].reshape(-1, 8, self.grid, self.grid)
        macro_id = state[:, -3]

        batch_size = state.shape[0]

        with torch.no_grad():
            self.orient_actor_net.eval()
            self.orient_critic_net.eval()
            self.place_actor_net.eval()
            self.place_critic_net.eval()

            orient_probs = self.orient_actor_net(canvas, wire_img_8oc, pos_mask_8oc)
            assert orient_probs.shape == (batch_size, 8), f"{orient_probs.shape=} != {(batch_size, 8)=}"
            orient_dist = Categorical(orient_probs)
            orient = orient_dist.sample()
            orient_log_prob = orient_dist.log_prob(orient)
            assert orient_log_prob.shape == (batch_size,), f"{orient_log_prob.shape=} != {(batch_size,)=}"
            orient_value = self.orient_critic_net(canvas, wire_img_8oc, pos_mask_8oc, macro_id)
            assert orient_value.shape == (batch_size, 1), f"{orient_value.shape=} != {(batch_size, 1)=}"

            wire_img_1oc = wire_img_8oc[:, orient.item(), :, :].reshape(-1, 1, self.grid, self.grid)
            pos_mask_1oc = pos_mask_8oc[:, orient.item(), :, :].reshape(-1, 1, self.grid, self.grid)

            action_probs = self.place_actor_net(canvas, wire_img_1oc, pos_mask_1oc)
            assert action_probs.shape == (batch_size, self.grid*self.grid), f"{action_probs.shape=} != {(batch_size, self.grid*self.grid)=}"
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            assert action_log_prob.shape == (batch_size,), f"{action_log_prob.shape=} != {(batch_size,)=}"
            action_value = self.place_critic_net(canvas, wire_img_1oc, pos_mask_1oc, macro_id, orient)
            assert action_value.shape == (batch_size,1), f"{action_value.shape=} != {(batch_size,1)=}"

            orient_info = orient.item(), orient_log_prob.item(), orient_value.item()
            action_info = action.item(), action_log_prob.item(), action_value.item()
            # 添加[0]来移除batch维度
            state_imgs = macro_id.long().item(), canvas[0].cpu().numpy(), wire_img_8oc[0].cpu().numpy(), pos_mask_8oc[0].cpu().numpy(), wire_img_1oc[0].cpu().numpy(), pos_mask_1oc[0].cpu().numpy()

        return orient_info, action_info, state_imgs
    
    def _update_train_flag(self):
        if self.placer_ok:
            if self.orient_agent_update_count is None and self.place_agent_update_count is None:
                # 第一次更新train flag
                self.orient_agent_update_count = 0
                self.place_agent_update_count = 0
                self.train_orient_agent = True
                self.train_place_agent = False
            else:
                if self.train_orient_agent:
                    self.orient_agent_update_count += 1
                    if self.orient_agent_update_count % 5 == 0:
                        self.train_orient_agent = False
                        self.train_place_agent = True
                elif self.train_place_agent:
                    self.place_agent_update_count += 1
                    if self.place_agent_update_count % 1 == 0:
                        self.train_place_agent = False
                        self.train_orient_agent = True
                else:
                    raise ValueError("orient, place train flag are False")
            print(f"update train flag, train_orient_agent={self.train_orient_agent}, train_place_agent={self.train_place_agent}")
            
    @trackit
    def update(self, data:dict=None) -> None:

        if data is None:
            # 单机模式，数据来自buffer
            target_list = []
            target = None
            for transition in reversed(self.buffer):
                if transition.done:
                    target = 0
                target = transition.reward + self.gamma * target
                target_list.append(target)
            target_list.reverse()
            target_v = torch.tensor(np.array(target_list), dtype=torch.float).view(-1, 1).to(self.device)
            state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
            orient = torch.tensor(np.array([t.orient for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
            action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
            old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
            old_orient_log_prob = torch.tensor(np.array([t.o_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
            self.buffer.clear()
        else:
            # 分布式训练，数据来自reverb
            macro_id = torch.tensor(data['macro_id'], dtype=torch.int64).to(self.device)
            canvas = torch.tensor(data['canvas'], dtype=torch.float).to(self.device)
            wire_img_8oc = torch.tensor(data['wire_img_8oc'], dtype=torch.float).to(self.device)
            pos_mask_8oc = torch.tensor(data['pos_mask_8oc'], dtype=torch.float).to(self.device)
            wire_img_1oc = torch.tensor(data['wire_img_1oc'], dtype=torch.float).to(self.device)
            pos_mask_1oc = torch.tensor(data['pos_mask_1oc'], dtype=torch.float).to(self.device)

            orient = torch.tensor(data['orient'], dtype=torch.int64).to(self.device)
            action = torch.tensor(data['action'], dtype=torch.int64).to(self.device)
            old_action_log_prob = torch.tensor(data['a_log_prob'], dtype=torch.float).to(self.device)
            old_orient_log_prob = torch.tensor(data['o_log_prob'], dtype=torch.float).to(self.device)
            orient_advantage = torch.tensor(data['o_advantage'], dtype=torch.float).to(self.device)
            action_advantage = torch.tensor(data['a_advantage'], dtype=torch.float).to(self.device)
            target_value = torch.tensor(data['return'], dtype=torch.float).to(self.device)
        

        action_advantage_mean = action_advantage.mean().cpu().item()
        action_advantage_std = action_advantage.std().cpu().item()
        action_advantage_pos_rate = (action_advantage > 0).float().mean().cpu().item()
        action_advantage_neg_rate = (action_advantage < 0).float().mean().cpu().item()
        logger.info(f"Action Advantage mean: {action_advantage_mean:.4f}, std: {action_advantage_std:.4f}, pos rate: {action_advantage_pos_rate*100:.2f}%, neg rate: {action_advantage_neg_rate*100:.2f}%")

        orient_advantage_mean = orient_advantage.mean().cpu().item()
        orient_advantage_std = orient_advantage.std().cpu().item()
        orient_advantage_pos_rate = (orient_advantage > 0).float().mean().cpu().item()
        orient_advantage_neg_rate = (orient_advantage < 0).float().mean().cpu().item()
        logger.info(f"Orient Advantage mean: {orient_advantage_mean:.4f}, std: {orient_advantage_std:.4f}, pos rate: {orient_advantage_pos_rate*100:.2f}%, neg rate: {orient_advantage_neg_rate*100:.2f}%")

        for epoch in range(self.ppo_epoch):  # iteration ppo_epoch
            epoch_progess = f" Epoch {epoch+1} / {self.ppo_epoch} "
            logger.info(f"{epoch_progess:-^80}")

            orient_ratio_list, orient_actor_losses, orient_critic_losses = [], [], []
            place_ratio_list, place_actor_losses, place_critic_losses = [], [], []
            for index in BatchSampler(SubsetRandomSampler(range(action.shape[0])), self.batch_size, True):
                self.training_step += 1

                batch_macro_id = macro_id[index].to(self.device)
                batch_canvas = canvas[index].to(self.device)
                batch_wire_img_8oc = wire_img_8oc[index].to(self.device)
                batch_pos_mask_8oc = pos_mask_8oc[index].to(self.device)
                batch_wire_img_1oc = wire_img_1oc[index].to(self.device)
                batch_pos_mask_1oc = pos_mask_1oc[index].to(self.device)

                state_dict = {
                    'macro_id': batch_macro_id,
                    'canvas': batch_canvas,
                    'wire_img_8oc': batch_wire_img_8oc,
                    'pos_mask_8oc': batch_pos_mask_8oc,
                    'wire_img_1oc': batch_wire_img_1oc,
                    'pos_mask_1oc': batch_pos_mask_1oc,
                }

                batch_orient = orient[index].to(self.device)
                batch_action = action[index].to(self.device)
                batch_old_orient_log_prob = old_orient_log_prob[index].to(self.device)
                batch_old_action_log_prob = old_action_log_prob[index].to(self.device)
                batch_orient_advantage = orient_advantage[index].to(self.device)
                batch_action_advantage = action_advantage[index].to(self.device)
                batch_target_value = target_value[index].to(self.device)

                if self.train_orient_agent:
                    ratios, actor_loss, critic_loss = self.update_orient_agent(state_dict, batch_orient, batch_target_value, batch_orient_advantage, batch_old_orient_log_prob)
                    orient_ratio_list.append(ratios)
                    orient_actor_losses.append(actor_loss)
                    orient_critic_losses.append(critic_loss)
                if self.train_place_agent:
                    ratios, actor_loss, critic_loss = self.update_place_agent(state_dict, batch_orient, batch_action, batch_target_value, batch_action_advantage, batch_old_action_log_prob)
                    place_ratio_list.append(ratios)
                    place_actor_losses.append(actor_loss)
                    place_critic_losses.append(critic_loss)

            if self.train_orient_agent:
                orient_ratio = np.concatenate(orient_ratio_list)
                orient_clip_rate = np.mean(np.abs(orient_ratio - 1) > self.clip_param)
                orient_up_rate = np.mean(orient_ratio > 1)
                orient_down_rate = np.mean(orient_ratio < 1)
                logger.info(f"orient_actor_loss: {np.mean(orient_actor_losses):.4e}, orient_critic_loss: {np.mean(orient_critic_losses):.4e}")
                logger.info(f"Orient ratio# clip_rate: {orient_clip_rate*100:.2f}%, up_rate: {orient_up_rate*100:.2f}%, down_rate: {orient_down_rate*100:.2f}%, max: {np.max(orient_ratio):.5f}, min: {np.min(orient_ratio):.5f}, mean: {np.mean(orient_ratio):.5f}, std: {np.std(orient_ratio):.5f}")

            if self.train_place_agent:
                place_ratio = np.concatenate(place_ratio_list)
                place_clip_rate = np.mean(np.abs(place_ratio - 1) > self.clip_param)
                place_up_rate = np.mean(place_ratio > 1)
                place_down_rate = np.mean(place_ratio < 1)
                logger.info(f"place_actor_loss: {np.mean(place_actor_losses):.4e}, place_critic_loss: {np.mean(place_critic_losses):.4e}")
                logger.info(f"Place ratio# clip_rate: {place_clip_rate*100:.2f}%, up_rate: {place_up_rate*100:.2f}%, down_rate: {place_down_rate*100:.2f}%, max: {np.max(place_ratio):.5f}, min: {np.min(place_ratio):.5f}, mean: {np.mean(place_ratio):.5f}, std: {np.std(place_ratio):.5f}")


        self._update_train_flag()

    def update_place_agent(self, state_dict, batch_orient, batch_action, batch_target_value, batch_action_advantage, batch_old_action_log_prob):
        canvas = state_dict['canvas']
        wire_img = state_dict['wire_img_1oc']
        pos_mask = state_dict['pos_mask_1oc']
        macro_id = state_dict['macro_id'].squeeze()
        orient = batch_orient.squeeze()
        batch_action = batch_action.squeeze()
        batch_action_advantage = batch_action_advantage.squeeze()
        batch_old_action_log_prob = batch_old_action_log_prob.squeeze()

        self.place_actor_net.train()
        self.place_critic_net.train()

        action_probs = self.place_actor_net(canvas, wire_img, pos_mask)
        action_log_prob = Categorical(action_probs).log_prob(batch_action)
        assert action_log_prob.shape == batch_old_action_log_prob.shape, f"{action_log_prob.shape=} != {batch_old_action_log_prob.shape=}"
        ratio = torch.exp(action_log_prob - batch_old_action_log_prob) # (batch_size,)

        normalize_advantage = (batch_action_advantage - batch_action_advantage.mean()) / (batch_action_advantage.std() + 1e-8)
        assert ratio.shape == normalize_advantage.shape, f"{ratio.shape=} != {normalize_advantage.shape=}"
        L1 = ratio * normalize_advantage
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * normalize_advantage
        assert L1.shape == L2.shape, f"{L1.shape=} != {L2.shape=}"
        action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
        assert action_loss.shape == (), f"{action_loss.shape=}"

        self.place_actor_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.place_actor_net.parameters(), self.max_grad_norm)
        self.place_actor_optimizer.step()
        # self.place_actor_scheduler.step()

        place_value = self.place_critic_net(canvas, wire_img, pos_mask, macro_id, orient)
        value_loss = F.smooth_l1_loss(place_value, batch_target_value)
        self.place_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.place_critic_net.parameters(), self.max_grad_norm)
        self.place_critic_optimizer.step()
        # self.place_critic_scheduler.step()

        return ratio.detach().cpu().numpy(), action_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()

    def update_orient_agent(self, state_dict, batch_orient, batch_target_value, batch_orient_advantage, batch_old_orient_log_prob):
        canvas = state_dict["canvas"]
        wire_img = state_dict["wire_img_8oc"]
        pos_mask = state_dict["pos_mask_8oc"]
        macro_id = state_dict["macro_id"].squeeze()
        batch_orient = batch_orient.squeeze()
        batch_orient_advantage = batch_orient_advantage.squeeze()
        batch_old_orient_log_prob = batch_old_orient_log_prob.squeeze()

        self.orient_actor_net.train()
        self.orient_critic_net.train()

        orient_probs = self.orient_actor_net(canvas, wire_img, pos_mask)
        orient_log_prob = Categorical(orient_probs).log_prob(batch_orient)
        assert orient_log_prob.shape == batch_old_orient_log_prob.shape, f"{orient_log_prob.shape=} != {batch_old_orient_log_prob.shape=}"
        ratio = torch.exp(orient_log_prob - batch_old_orient_log_prob)
        normalize_advantage = (batch_orient_advantage - batch_orient_advantage.mean()) / (batch_orient_advantage.std() + 1e-8)
        assert ratio.shape == normalize_advantage.shape, f"{ratio.shape=} != {normalize_advantage.shape=}"
        L1 = ratio * normalize_advantage
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * normalize_advantage
        assert L1.shape == L2.shape, f"{L1.shape=} != {L2.shape=}"
        orient_loss = -torch.min(L1, L2).mean()
        assert orient_loss.shape == (), f"{orient_loss.shape=}"

        self.orient_actor_optimizer.zero_grad()
        orient_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_actor_net.parameters(), self.max_grad_norm)
        self.orient_actor_optimizer.step()
        # self.orient_actor_scheduler.step()

        orient_value = self.orient_critic_net(canvas, wire_img, pos_mask, macro_id)
        value_loss = F.smooth_l1_loss(orient_value, batch_target_value)
        self.orient_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_critic_net.parameters(), self.max_grad_norm)
        self.orient_critic_optimizer.step()
        # self.orient_critic_scheduler.step()

        return ratio.detach().cpu().numpy(), orient_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()

