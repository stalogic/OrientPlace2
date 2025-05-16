import os
import sys
import gym
import time
import shutil
import argparse
from pathlib import Path
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import UniPPO
from util import set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--result_dir", type=str)
parser.add_argument("--clean", action="store_true", default=False, help="clean existing evaluation results")
parser.add_argument("--eval_num", type=int, default=10)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--cuda", type=str, default="")
args = parser.parse_args()

if args.seed is None:
    args.seed = int(time.time())
else:
    logger.info(f"Set seed to {args.seed}")
set_random_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

PROJECT_ROOT = Path(args.project_root)
data_root = PROJECT_ROOT / "benchmark"
cache_root = PROJECT_ROOT / "cache"
reader = LefDefReader(data_root, args.design_name, cache_root)
placedb = build_soft_macro_placedb(reader, cache_root=cache_root)
env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
placed_num_macro = len(placedb.macro_info)
agent = UniPPO(placed_num_macro, grid=224, batch_size=10, lr=3e-4, device='cuda')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE


def evaluate_model(chpt_path: Path):
    model_chpt = next((chpt_path / "checkpoints").glob("*_state_dict.pkl.gz"))
    agent.load_model(model_chpt)

    pic_path = chpt_path / "pictures"
    pl_path = chpt_path / "placements"

    pic_path.mkdir(parents=True, exist_ok=True)
    pl_path.mkdir(parents=True, exist_ok=True)

    hpwl_list = []

    for i in range(args.eval_num):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            if i == 0 and hasattr(agent, "greedy_action"):
                action = agent.greedy_action(state)
            else:
                action_info, _ = agent.select_action(state)
                action, _, _ = action_info
                
            orient = action // 224 // 224
            position = action % (224 * 224)
            next_state, reward, done, _ = env.step(position, orient)
            total_reward += reward
            state = next_state

        hpwl, cost = env.calc_hpwl_and_cost()
        hpwl_list.append(hpwl)
        prefix = f"H[{hpwl:.4e}_C[{cost:.4e}]_R[{total_reward:.4e}]]"
        env.plot(pic_path / f"evaluation_{i}_{prefix}.png")
        env.save_pl_file(pl_path / f"evaluation_{i}_{prefix}.pl")

    return min(hpwl_list), max(hpwl_list)


def eval():
    result_path = Path(args.result_dir)

    if args.clean:
        for p in result_path.iterdir():
            if p.is_dir():
                if (pic_path:=p / "pictures").exists():
                    shutil.rmtree(pic_path)
                if (pl_path:=p / "placements").exists():
                    shutil.rmtree(pl_path)
                if len(sps:=p.name.split("-")) >= 2:
                    p.rename(p.parent / "-".join(sps[:2]))

    while True:
        chpts:list[Path] = []
        for p in result_path.iterdir():
            if p.is_dir():
                sps = p.name.split("-")
                if len(sps) == 2:
                    chpts.append(p)
                elif "eval-failed" in p.name:
                    new_path = p.rename(p.parent / "-".join(sps[:2]))
                    chpts.append(new_path)

        if len(chpts) == 0:
            time.sleep(10)
            continue

        for p in sorted(chpts, key=lambda x: int(x.name.split("-")[1])):
            try:
                min_hpwl, max_hpwl = evaluate_model(p)
                p.rename(p.parent / f"{p.name}-{min_hpwl:.4e}-{max_hpwl:.4e}")
            except Exception as e:
                logger.info(f"{e}")
                p.rename(p.parent / f"{p.name}-eval-failed")


if __name__ == "__main__":
    eval()
