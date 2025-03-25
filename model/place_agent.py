import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from util import trackit

from .models import Actor, Critic


class PlacePPO:
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
        super(PlacePPO, self).__init__()
        self.placed_num_macro = placed_num_macro
        self.buffer_capacity = num_game_per_update * placed_num_macro
        self.grid = grid
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.actor_net = Actor().float().to(device)
        self.critic_net = Critic().float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr)

        self.CANVAS_SLICE = None
        self.WIRE_SLICE = None
        self.POS_SLICE = None
        self.FEATURE_SLICE = None

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self.actor_net.load_state_dict(checkpoint["actor_net_dict"])
        self.critic_net.load_state_dict(checkpoint["critic_net_dict"])

    def save_param(self, save_flag):
        torch.save(
            {
                "actor_net_dict": self.actor_net.state_dict(),
                "critic_net_dict": self.critic_net.state_dict(),
            },
            f"{save_flag}_net_dict.pkl",
        )

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    @trackit
    def select_action(self, state) -> tuple[int, int, float, float]:

        # in later version, orient will be choosed by OrientAgent
        orient = random.randint(0, 7)
        orient_prob = 1 / 8
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        canvas = state[:, self.CANVAS_SLICE].reshape(-1, 1, self.grid, self.grid)
        wire_img = state[:, self.WIRE_SLICE].reshape(-1, 8, self.grid, self.grid)
        wire_img = wire_img[:, orient, :, :].reshape(-1, 1, self.grid, self.grid)
        pos_mask = state[:, self.POS_SLICE].reshape(-1, 8, self.grid, self.grid)
        pos_mask = pos_mask[:, orient, :, :].reshape(-1, 1, self.grid, self.grid)

        with torch.no_grad():
            action_probs = self.actor_net(canvas, wire_img, pos_mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return orient, action.item(), orient_prob, action_log_prob.item()

    @trackit
    def update(self) -> None:

        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        orient = torch.tensor(np.array([t.orient for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
        old_orient_log_prob = torch.tensor(np.array([t.o_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)
        self.buffer.clear()

        target_list = []
        target = 0
        for i in range(reward.shape[0] - 1, -1, -1):
            if state[i, 0] >= self.placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(self.device)

        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                self.training_step += 1

                batch_state = state[index].to(self.device)

                canvas = batch_state[:, self.CANVAS_SLICE].reshape(self.batch_size, 1, self.grid, self.grid)
                wire_img = batch_state[:, self.WIRE_SLICE].reshape(self.batch_size * 8, self.grid, self.grid)
                pos_mask = batch_state[:, self.POS_SLICE].reshape(self.batch_size * 8, self.grid, self.grid)
                orient_index = orient[index].squeeze().long().cpu().tolist()
                orient_index = [batch * 8 + offset for batch, offset in enumerate(orient_index)]
                wire_img = wire_img[orient_index].reshape(self.batch_size, 1, self.grid, self.grid)
                pos_mask = pos_mask[orient_index].reshape(self.batch_size, 1, self.grid, self.grid)

                action_probs = self.actor_net(canvas, wire_img, pos_mask)
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze())
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())

                target_v = target_v_all[index]
                critic_net_output = self.critic_net(batch_state[:, -3], orient[index, 0])
                advantage = (target_v - critic_net_output).detach()

                L1 = ratio * advantage.squeeze()
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage.squeeze()
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(batch_state[:, -3], orient[index, 0]), target_v)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
