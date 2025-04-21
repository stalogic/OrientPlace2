import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from pathlib import Path
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

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
        lr: float,
        gamma: float,
        device: str,
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
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.place_actor_optimizer = optim.Adam(self.place_actor_net.parameters(), lr)
        self.place_critic_optimizer = optim.Adam(self.place_critic_net.parameters(), lr)
        self.orient_actor_optimizer = optim.Adam(self.orient_actor_net.parameters(), lr)
        self.orient_critic_optimizer = optim.Adam(self.orient_critic_net.parameters(), lr)
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

    def load_model(self, path: Path):
        with gzip.open(path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device(self.device))
            self.place_actor_net.load_state_dict(checkpoint["place_actor_net"])
            self.place_critic_net.load_state_dict(checkpoint["place_critic_net"])
            self.orient_actor_net.load_state_dict(checkpoint["orient_actor_net"])

    def save_model(self, save_path: Path, save_flag: str):
        save_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(save_path / f"{save_flag}_state_dict.pkl", "wb") as f:
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
    
    def debug_print(self):
        macro_id = torch.from_numpy(np.array([1])).long().to(self.device)
        canvas = torch.from_numpy(np.ones((1, 1, self.grid, self.grid), dtype=np.float32))
        wire_img_1oc = torch.from_numpy(np.ones((1, 1, self.grid, self.grid), dtype=np.float32))
        pos_mask_1oc = torch.from_numpy(np.ones((1, 1, self.grid, self.grid), dtype=np.float32))
        action_probs = self.place_actor_net(canvas, wire_img_1oc, pos_mask_1oc)
        action_value = self.place_critic_net(canvas, wire_img_1oc, pos_mask_1oc, macro_id)

        probs_sum = action_probs.squeeze().sum(dim=-1).item()
        logger.info(f"probs_sum: {probs_sum}, action_value: {action_value.squeeze().item()}, {action_probs[0,:10]=}")

    @trackit
    def select_action(self, state) -> tuple[int, int, float, float]:

        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        canvas = state[:, self.CANVAS_SLICE].reshape(-1, 1, self.grid, self.grid)
        wire_img_8oc = state[:, self.WIRE_SLICE].reshape(-1, 8, self.grid, self.grid)
        pos_mask_8oc = state[:, self.POS_SLICE].reshape(-1, 8, self.grid, self.grid)
        macro_id = state[:, -3]

        with torch.no_grad():
            # self.orient_actor_net.eval()
            # self.orient_critic_net.eval()
            # self.place_actor_net.eval()
            # self.place_critic_net.eval()
            self.debug_print()

            orient_probs = self.orient_actor_net(canvas, wire_img_8oc, pos_mask_8oc)
            orient_dist = Categorical(orient_probs)
            orient = orient_dist.sample()
            orient_prob = orient_dist.log_prob(orient)
            orient_value = self.orient_critic_net(canvas, wire_img_8oc, pos_mask_8oc, macro_id)

            wire_img_1oc = wire_img_8oc[:, orient.item(), :, :].reshape(-1, 1, self.grid, self.grid)
            pos_mask_1oc = pos_mask_8oc[:, orient.item(), :, :].reshape(-1, 1, self.grid, self.grid)

            action_probs = self.place_actor_net(canvas, wire_img_1oc, pos_mask_1oc)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            action_value = self.place_critic_net(canvas, wire_img_1oc, pos_mask_1oc, macro_id, orient)

            orient_info = orient.item(), orient_prob.item(), orient_value.item()
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

        self.debug_print()

        for epoch in range(self.ppo_epoch):  # iteration ppo_epoch
            logger.info(f"epoch {epoch+1} / {self.ppo_epoch}")
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
                    self.update_orient_agent(state_dict, batch_orient, batch_target_value, batch_orient_advantage, batch_old_orient_log_prob)
                if self.train_place_agent:
                    self.update_place_agent(state_dict, batch_orient, batch_action, batch_target_value, batch_action_advantage, batch_old_action_log_prob)

        self._update_train_flag()

    def update_place_agent(self, state_dict, batch_orient, batch_action, batch_target_value, batch_action_advantage, batch_old_action_log_prob):
        
        canvas = state_dict['canvas']
        wire_img = state_dict['wire_img_1oc']
        pos_mask = state_dict['pos_mask_1oc']
        macro_id = state_dict['macro_id'].squeeze()
        orient = batch_orient.squeeze()

        self.place_actor_net.train()
        self.place_critic_net.train()

        action_probs = self.place_actor_net(canvas, wire_img, pos_mask)
        dist = Categorical(action_probs)
        action_log_prob = dist.log_prob(batch_action)
        ratio = torch.exp(action_log_prob - batch_old_action_log_prob).squeeze()

        clip_rate = torch.abs(ratio - 1).gt(self.clip_param).float().mean()
        safe_rate = torch.logical_and(torch.gt(ratio, 1-self.clip_param), torch.lt(ratio, 1+self.clip_param)).float().mean()
        normalize_advantage = (batch_action_advantage - batch_action_advantage.mean()) / (batch_action_advantage.std() + 1e-8)
        logger.info(f"action clip rate: {clip_rate*100:.2f}%, safe rate: {safe_rate*100:.2f}%, max: {ratio.max()}, min: {ratio.min()}, mean: {ratio.mean()}, std: {ratio.std()}, advantage: {batch_action_advantage.abs().mean().item():.2f}, normalize advantage: {normalize_advantage.abs().mean().item():.2f}")

        L1 = ratio * normalize_advantage.squeeze()
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * normalize_advantage.squeeze()
        action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
        # logger.info(f"action policy loss: {action_loss.cpu().detach().numpy().item()}")

        self.place_actor_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.place_actor_net.parameters(), self.max_grad_norm)
        self.place_actor_optimizer.step()
        # self.place_actor_scheduler.step()

        place_value = self.place_critic_net(canvas, wire_img, pos_mask, macro_id, orient)
        value_loss = F.smooth_l1_loss(place_value, batch_target_value)
        # logger.info(f"action value loss: {value_loss.cpu().detach().numpy().item()}")
        self.place_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.place_critic_net.parameters(), self.max_grad_norm)
        self.place_critic_optimizer.step()
        # self.place_critic_scheduler.step()

    def update_orient_agent(self, batch_state, batch_orient, batch_target_value, batch_orient_advantage, batch_old_orient_log_prob):
        canvas = batch_state[:, self.CANVAS_SLICE].reshape(self.batch_size, 1, self.grid, self.grid)
        wire_img = batch_state[:, self.WIRE_SLICE].reshape(self.batch_size, 8, self.grid, self.grid)
        pos_mask = batch_state[:, self.POS_SLICE].reshape(self.batch_size, 8, self.grid, self.grid)
        macro_id = batch_state[:, -3]

        self.orient_actor_net.train()
        self.orient_critic_net.train()

        orient_probs = self.orient_actor_net(canvas, wire_img, pos_mask)
        dist = Categorical(orient_probs)
        orient_log_prob = dist.log_prob(batch_orient)
        ratio = torch.exp(orient_log_prob - batch_old_orient_log_prob).squeeze()

        clip_rate = torch.abs(ratio - 1).gt(self.clip_param).float().mean()
        normalize_advantage = (batch_orient_advantage - batch_orient_advantage.mean()) / (batch_orient_advantage.std() + 1e-8)
        logger.info(f"orient clip rate: {clip_rate*100:.2f}%, advantage: {batch_orient_advantage.abs().mean().item():.2f}, normalize advantage: {normalize_advantage.abs().mean().item():.2f}")

        L1 = ratio * normalize_advantage.squeeze()
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * normalize_advantage.squeeze()
        orient_loss = -torch.min(L1, L2).mean()
        logger.info(f"orient value loss: {orient_loss.cpu().detach().numpy().item()}")

        self.orient_actor_optimizer.zero_grad()
        orient_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_actor_net.parameters(), self.max_grad_norm)
        self.orient_actor_optimizer.step()
        # self.orient_actor_scheduler.step()

        orient_value = self.orient_critic_net(canvas, wire_img, pos_mask, macro_id)
        value_loss = F.smooth_l1_loss(orient_value, batch_target_value)
        logger.info(f"orient value loss: {value_loss.cpu().detach().numpy().item()}")
        self.orient_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_critic_net.parameters(), self.max_grad_norm)
        self.orient_critic_optimizer.step()
        # self.orient_critic_scheduler.step()

