import reverb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import gym
import tensorflow as tf
import time
import torch
import numpy as np
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
PROJECT_ROOT = "/home/jiangmingming/mntspace/OrientPlace2"
import random
from model import OrientPPO

data_root = os.path.join(PROJECT_ROOT, "benchmark")
design_name = "ariane133"
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, design_name, cache_root)
placedb = build_soft_macro_placedb(reader)

env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped


placed_num_macro = len(placedb.macro_info)
agent = OrientPPO(placed_num_macro, grid=224, num_game_per_update=5, batch_size=128, lr=1e-5, gamma=0.98, device='cuda')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE


def torch_to_tf(torch_tensor):
    """将 PyTorch 张量转换为 TensorFlow 张量"""
    numpy_array = torch_tensor.cpu().numpy()
    dtype_mapping = {
        torch.float32: tf.float32,
        torch.float64: tf.float64,
        torch.int32: tf.int32,
        torch.int64: tf.int64,
        torch.bool: tf.bool
    }
    tf_dtype = dtype_mapping.get(torch_tensor.dtype)
    if tf_dtype is None:
        raise ValueError(f"Unsupported PyTorch dtype: {torch_tensor.dtype}")
    return tf.convert_to_tensor(numpy_array, dtype=tf_dtype)

dataset = reverb.TrajectoryDataset.from_table_signature(
    server_address="localhost:12888",
    table='experience',
    max_in_flight_samples_per_worker=2
)

client = reverb.Client("localhost:12888")


for model_id in range(20):
    variables = {
        'orient_actor': agent.orient_actor_net.state_dict(),
        'place_actor': agent.place_actor_net.state_dict(),
        'model_id': torch.tensor(model_id, dtype=torch.int64)
    }
    variables = tf.nest.map_structure(torch_to_tf, variables)
    client.insert([variables], priorities={'model_info': model_id})
    
    episodes = []
    batch_size = 100
    t0 = time.time()
    discard = 0
    while len(episodes) < batch_size:
        size = min(5, batch_size-len(episodes))
        for sample in dataset.take(size):
            data_model_id = sample.data.get('model_id').numpy()
            if data_model_id == model_id:
                episodes.append(sample)
            else:
                discard += 1
    print(f"Time for collect {batch_size} samples {time.time()-t0:.2f}s, discard {discard} samples")
    
    time.sleep(10)
