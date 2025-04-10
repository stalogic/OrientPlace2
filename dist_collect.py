import gym
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import OrientPPO
import reverb
import torch
import numpy as np
import tensorflow as tf
import time

PROJECT_ROOT = "/home/jiangmingming/mntspace/OrientPlace2"


client = reverb.Client("localhost:12888")


data_root = os.path.join(PROJECT_ROOT, "benchmark")
design_name = "ariane133"
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, design_name, cache_root)
placedb = build_soft_macro_placedb(reader)

env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
agent = OrientPPO(len(placedb.macro_info), grid=224, num_game_per_update=5, batch_size=128, lr=1e-5, gamma=0.98, device='cuda')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE

def tf_to_torch(tf_tensor):
    """将 TensorFlow 张量转换为 PyTorch 张量"""
    numpy_array = tf_tensor.numpy()
    dtype_mapping = {
        tf.float32: torch.float32,
        tf.float64: torch.float64,
        tf.int32: torch.int32,
        tf.int64: torch.int64,
        tf.bool: torch.bool
    }
    torch_dtype = dtype_mapping.get(tf_tensor.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported TensorFlow dtype: {tf_tensor.dtype}")
    return torch.tensor(numpy_array, dtype=torch_dtype)

model_info = reverb.TimestepDataset.from_table_signature(
    server_address="localhost:12888",
    table="model_info",
    max_in_flight_samples_per_worker=1
)

while True:
    t0 = time.time()
    model = next(iter(model_info.take(1)))
    print(f"Read model from reverb server: {time.time() - t0:.2f}s")
    t0 = time.time()
    model_data = model.data
    model_data = tf.nest.map_structure(tf_to_torch, model_data)
    model_id = model_data.pop('model_id').numpy().item()
    print(f"{model_id=}")
    orient_actor = model_data['orient_actor']
    agent.orient_actor_net.load_state_dict(orient_actor)
    place_actor = model_data['place_actor']
    agent.place_actor_net.load_state_dict(place_actor)
    print(f"Load model state dict: {time.time() - t0:.2f}s")

    state = env.reset()
    done = False
    with client.trajectory_writer(num_keep_alive_refs=200) as writer:
        t0 = time.time()
        while not done:
            orient, action, orient_log_prob, action_log_prob = agent.select_action(state)

            next_state, reward, done, info = env.step(action, orient)

            writer.append({
                'state': state,
                'orient': orient,
                'action': action,
                'o_log_prob': orient_log_prob,
                'a_log_prob': action_log_prob,
                'reward': float(reward),
                'next_state': next_state,
                'done': done,
                'model_id': model_id,
            })
            state = next_state
        print(f"game time: {time.time() - t0:.2f}s")
        t0 = time.time()
        writer.create_item('experience', model_id, 
                        trajectory= {
                            'state': writer.history['state'][:],
                            'orient': writer.history['orient'][:],
                            'action': writer.history['action'][:],
                            'o_log_prob': writer.history['o_log_prob'][:],
                            'a_log_prob': writer.history['a_log_prob'][:],
                            'reward': writer.history['reward'][:],
                            'next_state': writer.history['next_state'][:],
                            'done': writer.history['done'][:],
                            'model_id': writer.history['model_id'][-1]
                        })
        writer.flush()
        print(f"Push trajectory {time.time() - t0:.2f}s")
