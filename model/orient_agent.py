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

    @trackit
    def select_action(self, state) -> tuple[int, int, float, float]:

        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        canvas = state[:, self.CANVAS_SLICE].reshape(-1, 1, self.grid, self.grid)
        wire_img = state[:, self.WIRE_SLICE].reshape(-1, 8, self.grid, self.grid)
        pos_mask = state[:, self.POS_SLICE].reshape(-1, 8, self.grid, self.grid)

        with torch.no_grad():
            self.orient_actor_net.eval()
            self.place_actor_net.eval()

            orient_probs = self.orient_actor_net(canvas, wire_img, pos_mask)
            orient_dist = Categorical(orient_probs)
            orient = orient_dist.sample()
            orient_prob = orient_dist.log_prob(orient)

            wire_img = wire_img[:, orient.item(), :, :].reshape(-1, 1, self.grid, self.grid)
            pos_mask = pos_mask[:, orient.item(), :, :].reshape(-1, 1, self.grid, self.grid)

            action_probs = self.place_actor_net(canvas, wire_img, pos_mask)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

            self.orient_actor_net.train()
            self.place_actor_net.train()
        return orient.item(), action.item(), orient_prob.item(), action_log_prob.item()
    
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
            target_v = torch.tensor(data['return'], dtype=torch.float).to(self.device)
            state = torch.tensor(data['state'], dtype=torch.float).to(self.device)
            orient = torch.tensor(data['orient'], dtype=torch.float).to(self.device)
            action = torch.tensor(data['action'], dtype=torch.float).to(self.device)
            old_action_log_prob = torch.tensor(data['a_log_prob'], dtype=torch.float).to(self.device)
            old_orient_log_prob = torch.tensor(data['o_log_prob'], dtype=torch.float).to(self.device)

        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(target_v.shape[0])), self.batch_size, True):
                self.training_step += 1
                batch_state = state[index].to(self.device)
                batch_orient = orient[index].to(self.device)
                batch_action = action[index].to(self.device)
                batch_target = target_v[index].to(self.device)
                batch_old_orient_log_prob = old_orient_log_prob[index].to(self.device)
                batch_old_action_log_prob = old_action_log_prob[index].to(self.device)

                if self.train_place_agent:
                    self.update_place_agent(batch_state, batch_orient, batch_action, batch_target, batch_old_action_log_prob)
                if self.train_orient_agent:
                    self.update_orient_agent(batch_state, batch_orient, batch_target, batch_old_orient_log_prob)

        self._update_train_flag()

    def update_place_agent(self, batch_state, batch_orient, batch_action, batch_target, batch_old_action_log_prob):
        canvas = batch_state[:, self.CANVAS_SLICE].reshape(self.batch_size, 1, self.grid, self.grid)
        wire_img = batch_state[:, self.WIRE_SLICE].reshape(self.batch_size * 8, self.grid, self.grid)
        pos_mask = batch_state[:, self.POS_SLICE].reshape(self.batch_size * 8, self.grid, self.grid)
        orient_index = batch_orient.squeeze().long().cpu().tolist()
        orient_index = [batch_id * 8 + offset for batch_id, offset in enumerate(orient_index)]
        wire_img = wire_img[orient_index].reshape(self.batch_size, 1, self.grid, self.grid)
        pos_mask = pos_mask[orient_index].reshape(self.batch_size, 1, self.grid, self.grid)

        action_probs = self.place_actor_net(canvas, wire_img, pos_mask)
        dist = Categorical(action_probs)
        action_log_prob = dist.log_prob(batch_action.squeeze())
        ratio = torch.exp(action_log_prob - batch_old_action_log_prob.squeeze())

        with torch.no_grad():
            self.place_critic_net.eval()
            critic_net_output = self.place_critic_net(canvas, wire_img, pos_mask, batch_state[:, -3], batch_orient[:, 0])
            advantage = (batch_target - critic_net_output).detach()
            self.place_critic_net.train()

        L1 = ratio * advantage.squeeze()
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage.squeeze()
        action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
        logger.info(f"action policy loss: {action_loss.cpu().detach().numpy().item()}")

        self.place_actor_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.place_actor_net.parameters(), self.max_grad_norm)
        self.place_actor_optimizer.step()
        self.place_actor_scheduler.step()

        value_loss = F.smooth_l1_loss(self.place_critic_net(canvas, wire_img, pos_mask, batch_state[:, -3], batch_orient[:, 0]), batch_target)
        logger.info(f"action value loss: {value_loss.cpu().detach().numpy().item()}")
        self.place_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.place_critic_net.parameters(), self.max_grad_norm)
        self.place_critic_optimizer.step()
        self.place_critic_scheduler.step()

    def update_orient_agent(self, batch_state, batch_orient, batch_target, batch_old_orient_log_prob):
        canvas = batch_state[:, self.CANVAS_SLICE].reshape(self.batch_size, 1, self.grid, self.grid)
        wire_img = batch_state[:, self.WIRE_SLICE].reshape(self.batch_size, 8, self.grid, self.grid)
        pos_mask = batch_state[:, self.POS_SLICE].reshape(self.batch_size, 8, self.grid, self.grid)

        orient_probs = self.orient_actor_net(canvas, wire_img, pos_mask)
        dist = Categorical(orient_probs)
        orient_log_prob = dist.log_prob(batch_orient.squeeze())
        ratio = torch.exp(orient_log_prob - batch_old_orient_log_prob.squeeze())

        with torch.no_grad():
            self.orient_critic_net.eval()
            critic_net_output = self.orient_critic_net(canvas, wire_img, pos_mask, batch_state[:, -3])
            advantage = (batch_target - critic_net_output).detach()
            self.orient_critic_net.train()

        L1 = ratio * advantage.squeeze()
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage.squeeze()
        orient_loss = -torch.min(L1, L2).mean()
        logger.info(f"orient value loss: {orient_loss.cpu().detach().numpy().item()}")

        self.orient_actor_optimizer.zero_grad()
        orient_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_actor_net.parameters(), self.max_grad_norm)
        self.orient_actor_optimizer.step()
        self.orient_actor_scheduler.step()

        value_loss = F.smooth_l1_loss(self.orient_critic_net(canvas, wire_img, pos_mask, batch_state[:, -3]), batch_target)
        logger.info(f"orient value loss: {value_loss.cpu().detach().numpy().item()}")
        self.orient_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_critic_net.parameters(), self.max_grad_norm)
        self.orient_critic_optimizer.step()
        self.orient_critic_scheduler.step()

