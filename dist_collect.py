import gym
import os
import place_env
from placedb import LefDefReader, build_soft_macro_placedb

import reverb

PROJECT_ROOT = "/home/jiangmingming/mntspace/OrientPlace2"


client = reverb.Client("localhost:12888")


data_root = os.path.join(PROJECT_ROOT, "benchmark")
design_name = "ariane133"
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, design_name, cache_root)
placedb = build_soft_macro_placedb(reader)

env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped

with client.trajectory_writer(num_keep_alive_refs=200) as writer:
    while True:
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            action_log_prob = 1e-3
            orient = env.orient_space.sample()
            orient_log_prob = 1e-1

            next_state, reward, done, info = env.step(action, orient)

            writer.append({
                'state': state,
                'orient': orient,
                'action': action,
                'o_log_prob': orient_log_prob,
                'a_log_prob': action_log_prob,
                'reward': float(reward),
                'next_state': next_state,
                'done': done
            })
            state = next_state
        writer.create_item('experience', 1.0, 
                        trajectory= {
                            'state': writer.history['state'][:],
                            'orient': writer.history['orient'][:],
                            'action': writer.history['action'][:],
                            'o_log_prob': writer.history['o_log_prob'][:],
                                'a_log_prob': writer.history['a_log_prob'][:],
                                'reward': writer.history['reward'][:],
                                'next_state': writer.history['next_state'][:],
                                'done': writer.history['done'][:]
                        })
