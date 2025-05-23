import os
import sys
import gym
import torch
import reverb
import argparse
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import UniPPO
from reverb_util import to_tf_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--reverb_batch", type=int, default=10)

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
agent = UniPPO(placed_num_macro, grid=224, batch_size=128, lr=1e-5, device='cpu')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE

model_variables = {
    'uni_actor': agent.uni_actor_net.state_dict(),
    'uni_critic': agent.uni_critic_net.state_dict(),
    'model_id': torch.tensor(0, dtype=torch.int64)
}

model_signature = tf.nest.map_structure(
    lambda var: tf.TensorSpec(var.shape, to_tf_dtype(var.dtype)),
    model_variables
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
    'macro_id': tf.TensorSpec(shape=[None], dtype=tf.int64),
    'state': tf.TensorSpec(shape=[None, 9, 224, 224], dtype=tf.float32),
    'mask': tf.TensorSpec(shape=[None, 8, 224, 224], dtype=tf.float32),

    'action': tf.TensorSpec(shape=[None], dtype=tf.int64),
    'reward': tf.TensorSpec(shape=[None], dtype=tf.float32),
    'log_prob': tf.TensorSpec(shape=[None], dtype=tf.float32),
    'value': tf.TensorSpec(shape=[None], dtype=tf.float32),
    'return': tf.TensorSpec(shape=[None], dtype=tf.float32),
    'advantage': tf.TensorSpec(shape=[None], dtype=tf.float32),
    'model_id': tf.TensorSpec(shape=[], dtype=tf.int64),
}

data_table = reverb.Table(
    name="experience",
    sampler=reverb.selectors.MaxHeap(),
    remover=reverb.selectors.MinHeap(),
    max_size=20,
    rate_limiter=reverb.rate_limiters.SampleToInsertRatio(1, args.reverb_batch, [args.reverb_batch, args.reverb_batch*2]),
    max_times_sampled=1,
    signature=data_signature
)

server = reverb.Server([model_table, data_table], port=args.reverb_port)
del env
del agent
del placedb
server.wait()