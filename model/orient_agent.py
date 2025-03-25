import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

        self.CANVAS_SLICE = None
        self.WIRE_SLICE = None
        self.POS_SLICE = None
        self.FEATURE_SLICE = None

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self.place_actor_net.load_state_dict(checkpoint["actor_net_dict"])
        self.place_critic_net.load_state_dict(checkpoint["critic_net_dict"])

    def save_param(self, save_flag):
        torch.save(
            {
                "actor_net_dict": self.place_actor_net.state_dict(),
                "critic_net_dict": self.place_critic_net.state_dict(),
            },
            f"{save_flag}_net_dict.pkl",
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
        return orient.item(), action.item(), orient_prob.item(), action_log_prob.item()

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
                batch_orient = orient[index].to(self.device)
                batch_action = action[index].to(self.device)
                batch_target = target_v_all[index].to(self.device)
                batch_old_orient_log_prob = old_orient_log_prob[index].to(self.device)
                batch_old_action_log_prob = old_action_log_prob[index].to(self.device)

                self.update_place_agent(batch_state, batch_orient, batch_action, batch_target, batch_old_action_log_prob)
                self.update_orient_agent(batch_state, batch_orient, batch_target, batch_old_orient_log_prob)
                

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

        critic_net_output = self.place_critic_net(batch_state[:, -3], batch_orient[:, 0])
        advantage = (batch_target - critic_net_output).detach()

        L1 = ratio * advantage.squeeze()
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage.squeeze()
        action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent

        self.place_actor_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.place_actor_net.parameters(), self.max_grad_norm)
        self.place_actor_optimizer.step()

        value_loss = F.smooth_l1_loss(self.place_critic_net(batch_state[:, -3], batch_orient[:, 0]), batch_target)
        self.place_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.place_critic_net.parameters(), self.max_grad_norm)
        self.place_critic_optimizer.step()


    def update_orient_agent(self, batch_state, batch_orient, batch_target, batch_old_orient_log_prob):
        canvas = batch_state[:, self.CANVAS_SLICE].reshape(self.batch_size, 1, self.grid, self.grid)
        wire_img = batch_state[:, self.WIRE_SLICE].reshape(self.batch_size, 8, self.grid, self.grid)
        pos_mask = batch_state[:, self.POS_SLICE].reshape(self.batch_size, 8, self.grid, self.grid)

        orient_probs = self.orient_actor_net(canvas, wire_img, pos_mask)
        dist = Categorical(orient_probs)
        orient_log_prob = dist.log_prob(batch_orient.squeeze())
        ratio = torch.exp(orient_log_prob - batch_old_orient_log_prob.squeeze())

        critic_net_output = self.orient_critic_net(batch_state[:, -3])
        advantage = (batch_target - critic_net_output).detach()

        L1 = ratio * advantage.squeeze()
        L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage.squeeze()
        orient_loss = -torch.min(L1, L2).mean()

        self.orient_actor_optimizer.zero_grad()
        orient_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_actor_net.parameters(), self.max_grad_norm)
        self.orient_actor_optimizer.step()

        value_loss = F.smooth_l1_loss(self.orient_critic_net(batch_state[:, -3]), batch_target)
        self.orient_critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.orient_critic_net.parameters(), self.max_grad_norm)
        self.orient_critic_optimizer.step()

