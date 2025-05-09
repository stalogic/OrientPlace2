import os
import gym
import time
import torch
import reverb
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import tensorflow as tf

from pathlib import Path
from loguru import logger

import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import UniPPO
from reverb_util import torch_to_tf, PlaceTrajectoryDataset
from util import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--mini_batch", type=int, default=128)
parser.add_argument("--reverb_batch", type=int, default=10)
parser.add_argument("--model_iterations", type=int, default=2000, help="model iteration")
parser.add_argument("--iter_per_model", type=int, default=10, help="iteration per model")

parser.add_argument("--init_chpt", type=Path, default=None, help="init chpt path")

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reverb_ip", type=str, default="localhost")
parser.add_argument("--reverb_port", type=int, default=12888)
parser.add_argument("--cuda", type=str, default="")
args = parser.parse_args()

if args.seed is None:
    args.seed = int(time.time())
set_random_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

PROJECT_ROOT = Path(args.project_root)
data_root = PROJECT_ROOT / "benchmark"
cache_root = PROJECT_ROOT / "cache"
reader = LefDefReader(data_root, args.design_name, cache_root)
placedb = build_soft_macro_placedb(reader, cache_root=cache_root)
env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
placed_num_macro = len(placedb.macro_info)

strftime = time.strftime("%m%d%H%M", time.localtime())
run_id = f"{args.design_name}_{strftime}_pnm_{placed_num_macro}_seed_{args.seed}"
RESULT_ROOT = PROJECT_ROOT / "result" / run_id
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

REVERB_ADDR = f"{args.reverb_ip}:{args.reverb_port}"

dataset = reverb.TrajectoryDataset.from_table_signature(
    server_address=REVERB_ADDR,
    table='experience',
    max_in_flight_samples_per_worker=args.reverb_batch,
).batch(args.reverb_batch)

batch_reader = PlaceTrajectoryDataset(dataset, batch_size=args.reverb_batch)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def init_agent(rank):

    learning_rate = {
        "uni_actor": 1e-5,
        "uni_critic": 1e-4,
    }
    agent = UniPPO(placed_num_macro, grid=224, batch_size=args.mini_batch, lr=learning_rate, device=f"cuda:{rank}", ddp=True)
    agent.CANVAS_SLICE = env.CANVAS_SLICE
    agent.WIRE_SLICE = env.WIRE_SLICE
    agent.POS_SLICE = env.POS_SLICE
    agent.FEATURE_SLICE = env.FEATURE_SLICE

    if args.init_chpt is not None:
        agent.load_model(args.init_chpt)

    return agent

def train(rank, world_size):
    setup(rank, world_size)

    agent = init_agent(rank)

    client = reverb.Client(REVERB_ADDR)
    logger.info(f"start training, model_iterations: {args.model_iterations}, iter_per_model: {args.iter_per_model}")
    for model_id in range(args.model_iterations):
        t0 = time.time()
        if rank == 0:
            agent.save_model(RESULT_ROOT / f"chpt-{model_id}" / "checkpoints", f"M{model_id}")
            model_variables = {
                'uni_actor': agent.uni_actor_net.state_dict(),
                'uni_critic': agent.uni_critic_net.state_dict(),
                'model_id': torch.tensor(model_id, dtype=torch.int64)
            }
            model_variables = tf.nest.map_structure(torch_to_tf, model_variables)
            client.insert([model_variables], priorities={'model_info': model_id})
        save_time = time.time() - t0
        
        if rank == 0:
            model_version_id = torch.tensor([model_id], dtype=torch.int64).to(rank)
        else:
            model_version_id = torch.empty(1, dtype=torch.int64).to(rank)
        dist.broadcast(model_version_id, src=0)

        data_time, update_time = 0, 0
        for i in range(args.iter_per_model):
            t0 = time.time()
            data = batch_reader.read(model_id=model_version_id.item())
            t1 = time.time()
            agent.update(data)
            t2 = time.time()
            data_time += t1 - t0
            update_time += t2 - t1
            logger.info(f"Rank: {rank}, iter: {i+1}/{args.iter_per_model}, model_id: {model_id}, data_time: {t1 - t0:.3f}, update_time: {t2 - t1:.3f}")

        logger.info(f"Rank: {rank}, model_id: {model_id} finished, save_time: {save_time:.3f} data_time: {data_time:.3f}, update_time: {update_time:.3f}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    