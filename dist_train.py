import os
import gym
import time
import torch
import reverb
import argparse
import tensorflow as tf
from loguru import logger

from OrientPlace2.dist_collect import REVERB_ADDR
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import OrientPPO
from reverb_util import torch_to_tf
from util import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--mini_batch", type=int, default=128)
parser.add_argument("--reverb_batch", type=int, default=10)
parser.add_argument("--model_iterations", type=int, default=2000, help="model iteration")

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reverb_ip", type=str, default="localhost")
parser.add_argument("--reverb_port", type=int, default=12888)
parser.add_argument("--cuda", type=str, default="")
args = parser.parse_args()

if args.seed is None:
    args.seed = int(time.time())
set_random_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

PROJECT_ROOT = args.project_root
data_root = os.path.join(PROJECT_ROOT, "benchmark")
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, args.design_name, cache_root)
placedb = build_soft_macro_placedb(reader)
env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
placed_num_macro = len(placedb.macro_info)
agent = OrientPPO(placed_num_macro, grid=224, batch_size=args.mini_batch, lr=1e-5, gamma=0.98, device='cuda')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE

REVERB_ADDR = f"{args.reverb_ip}:{args.reverb_port}"

dataset = reverb.TrajectoryDataset.from_table_signature(
    server_address=REVERB_ADDR,
    table='experience',
    max_in_flight_samples_per_worker=2
).batch(args.reverb_batch)

client = reverb.Client(REVERB_ADDR)


for model_id in range(args.model_iterations):
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
