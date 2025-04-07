import gym
import os
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import OrientPPO
import reverb

PROJECT_ROOT = "/home/jiangmingming/mntspace/OrientPlace2"


client = reverb.Client("localhost:12888")


data_root = os.path.join(PROJECT_ROOT, "benchmark")
design_name = "ariane133"
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, design_name, cache_root)
placedb = build_soft_macro_placedb(reader)

env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
agent = OrientPPO(len(placedb.macro_info), grid=224, num_game_per_update=5, batch_size=128, lr=1e-5, gamma=0.98, device='cpu')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE


model_info = reverb.TimestepDataset.from_table_signature(
    server_address="localhost:12888",
    table="model_info",
    max_in_flight_samples_per_worker=1
)


with client.trajectory_writer(num_keep_alive_refs=200) as writer:
    while True:
        for model in model_info.take(1):
            for key, value in model.data.items():
                # print(f"{key=}, {type(value)}")
                if key == "model_id":
                    print(f"{key=}, {value.numpy().item()=}")
                else:
                    print(f"{key=}, {len(value)=}")

        continue
        
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
