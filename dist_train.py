import os
import gym
import time
import torch
import reverb
import argparse
import numpy as np
import tensorflow as tf
from loguru import logger
from collections import defaultdict

import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import OrientPPO
from reverb_util import torch_to_tf
from util import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--mini_batch", type=int, default=128)
parser.add_argument("--reverb_batch", type=int, default=5)
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
placedb = build_soft_macro_placedb(reader, cache_root=cache_root)
env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
placed_num_macro = len(placedb.macro_info)
agent = OrientPPO(placed_num_macro, grid=224, num_game_per_update=10, batch_size=args.mini_batch, lr=1e-5, gamma=0.98, device='cuda')
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

def get_batch_generator(dataset: reverb.TrajectoryDataset, batch_size: int, info: dict):
    batches = defaultdict(list)
    while True:
        batch = next(iter(dataset.take(1)))
        model_id = info["model_id"]
        print(f"model_id: {model_id}")
        batch = tf.nest.map_structure(lambda x: x.numpy(), batch.data)
        model_ids = batch.pop('model_id')
        print(f"{model_ids.shape=} {model_ids}")
        
        valid_mask = model_ids == model_id
        batch = tf.nest.map_structure(lambda x: x[valid_mask], batch)
        print(f"valid batch: {len(batch['state'])}, {valid_mask=}")
        for i in range(len(batch['state'])):
            for key, value in batch.items():
                batches[key].append(value[i])
            if len(batches[key]) >= batch_size:
                for key, value in batches.items():
                    print(f"{key}: {len(value)}, {value[0].shape}")
                
                result = {}
                for key, value in batches.items():
                    value = np.concatenate(value, axis=0)
                    if len(value.shape) == 1:
                        value = np.expand_dims(value, axis=1)
                    elif len(value.shape) > 2:
                        raise ValueError(f"Invalid shape: {value.shape} for `{key}`")
                    result[key] = value
                yield result
                batches.clear()

info = {'model_id': 0}
batch_generator = get_batch_generator(dataset, 10, info)

def train():

    client = reverb.Client(REVERB_ADDR)

    for model_id in range(args.model_iterations):
        variables = {
            'orient_actor': agent.orient_actor_net.state_dict(),
            'place_actor': agent.place_actor_net.state_dict(),
            'model_id': torch.tensor(model_id, dtype=torch.int64)
        }
        variables = tf.nest.map_structure(torch_to_tf, variables)
        client.insert([variables], priorities={'model_info': model_id})

        info["model_id"] = model_id
        t0 = time.time()
        for _ in range(10):
            data = next(batch_generator)
            agent.update(data)

        logger.info(f"model_id: {model_id} trained, time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    train()