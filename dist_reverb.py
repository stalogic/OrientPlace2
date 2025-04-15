import os
import gym
import torch
import reverb
import argparse
import tensorflow as tf

import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import OrientPPO
from reverb_util import to_tf_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reverb_ip", type=str, default="localhost")
parser.add_argument("--reverb_port", type=int, default=12888)
parser.add_argument("--cuda", type=str, default="")
args = parser.parse_args()

PROJECT_ROOT = args.project_root
data_root = os.path.join(PROJECT_ROOT, "benchmark")
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, args.design_name, cache_root)
placedb = build_soft_macro_placedb(reader, cache_root=cache_root)
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
    'model_id': torch.tensor(0, dtype=torch.int64)
}

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
    'done': tf.TensorSpec(shape=[None], dtype=tf.bool),
    'return': tf.TensorSpec(shape=[None], dtype=tf.float64),
    'model_id': tf.TensorSpec(shape=[], dtype=tf.int64),
}

data_table = reverb.Table(
    name="experience",
    sampler=reverb.selectors.MaxHeap(),
    remover=reverb.selectors.MinHeap(),
    max_size=200,
    rate_limiter=reverb.rate_limiters.SampleToInsertRatio(1, 5, [5, 10]),
    max_times_sampled=1,
    signature=data_signature
)

server = reverb.Server([model_table, data_table], port=args.reverb_port)
del env
del agent
del placedb
server.wait()