import os
import gym
import time
import argparse
from pathlib import Path
from loguru import logger

import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import OrientPPO
from util import set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--result_dir", type=str)
parser.add_argument("--eval_num", type=int, default=10)

parser.add_argument("--seed", type=int, default=None)
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
agent = OrientPPO(placed_num_macro, grid=224, num_game_per_update=10, batch_size=args.mini_batch, lr=3e-4, gamma=0.98, device='cuda')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE


def evaluate_model(chpt_path: Path):
    model_chpt = next((chpt_path / "checkpoints").glob("*_state_dict.pkl.gz"))
    agent.load_model(model_chpt)

    for i in range(args.eval_num):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            orient_info, action_info, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_info[0], orient_info[0])
            total_reward += reward
            state = next_state

        hpwl, cost = env.calc_hpwl_and_cost()
        env.plot(chpt_path / "pictures" / f"evaluation_{i}_h{hpwl}_c{cost}_r{total_reward}.png")
        env.save_pl_file(chpt_path / "placements" / f"evaluation_{i}_h{hpwl}_c{cost}_r{total_reward}.pl")


def eval():
    eval_set = set()
    result_path = Path(args.result_dir)
    while True:
        for p in result_path.iterdir():
            if not p.is_dir():
                continue

            model_id = int(p.name.split("_")[1])
            if model_id in eval_set:
                continue
            eval_set.add(model_id)
            evaluate_model(p)


if __name__ == "__main__":
    eval()
