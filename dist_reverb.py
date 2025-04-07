import reverb
import os
import gym
import tensorflow as tf
import time
import torch
import numpy as np
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
PROJECT_ROOT = "/home/jiangmingming/mntspace/OrientPlace2"

from model import OrientPPO

data_root = os.path.join(PROJECT_ROOT, "benchmark")
design_name = "ariane133"
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, design_name, cache_root)
placedb = build_soft_macro_placedb(reader)

env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped


placed_num_macro = len(placedb.macro_info)
agent = OrientPPO(placed_num_macro, grid=224, num_game_per_update=5, batch_size=128, lr=1e-5, gamma=0.98, device='cpu')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE

variables = {
    'orient_actor': agent.orient_actor_net.state_dict(),
    'place_actor': agent.place_actor_net.state_dict(),
    'model_id': torch.from_numpy(np.array(0, dtype=np.int32))
}

def to_tf_dtype(torch_dtype):
    dtype_mapping = {
        torch.float32: tf.float32,
        torch.float64: tf.float64,
        torch.int32: tf.int32,
        torch.int64: tf.int64,
        torch.bool: tf.bool
    }
    return dtype_mapping[torch_dtype]

model_signature = tf.nest.map_structure(
    lambda var: tf.TensorSpec(var.shape, to_tf_dtype(var.dtype)),
    variables
)

model_table = reverb.Table(
    name="model_info",
    sampler=reverb.selectors.Fifo(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    max_size=1,
    max_times_sampled=0,
    signature=model_signature
)

data_signature = {
    'state': tf.TensorSpec(shape=[None, 852995], dtype=tf.float64),
    'orient': tf.TensorSpec(shape=[None], dtype=tf.int64),
    'action': tf.TensorSpec(shape=[None], dtype=tf.int64),
    'o_log_prob': tf.TensorSpec(shape=[None], dtype=tf.float64),
    'a_log_prob': tf.TensorSpec(shape=[None], dtype=tf.float64),
    'reward': tf.TensorSpec(shape=[None], dtype=tf.float64),
    'next_state': tf.TensorSpec(shape=[None, 852995], dtype=tf.float64),
    'done': tf.TensorSpec(shape=[None], dtype=tf.bool)
}

data_table = reverb.Table(
    name="experience",
    sampler=reverb.selectors.MaxHeap(),
    remover=reverb.selectors.MinHeap(),
    max_size=10000,
    rate_limiter=reverb.rate_limiters.MinSize(5),
    signature=data_signature
)

server = reverb.Server([model_table, data_table], port=12888)
dataset = reverb.TrajectoryDataset.from_table_signature(
    server_address="localhost:12888",
    table='experience',
    max_in_flight_samples_per_worker=2
)

client = reverb.Client("localhost:12888")



for _ in range(10):
    variables = {
        'orient_actor': agent.orient_actor_net.state_dict(),
        'place_actor': agent.place_actor_net.state_dict(),
        'model_id': variables['model_id'] + 1
    }
    client.insert([variables], priorities={'model_info': variables['model_id'].numpy().item()})

    for sample in dataset.take(2):
        # print(sample.info)
        for key, value in sample.data.items():
            print(f"{key=} shape: {value.shape} dtype: {value.dtype}, numpy.shape: {value.numpy().shape}")
    time.sleep(100)







# # 关闭服务器
server.stop()