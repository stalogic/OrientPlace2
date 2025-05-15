import argparse
import os
import pathlib
import time
import gym
import torch
import numpy as np
from collections import defaultdict

import place_env
from model import OrientPPO
from placedb import build_soft_macro_placedb, get_design_reader
from util import set_random_seed


# Parameters
parser = argparse.ArgumentParser(description="Solve Macro Placement Task with PPO")
# design data parameters
parser.add_argument("--data_root", default="./benchmark/", help="the parent dir of innovus workspace")
parser.add_argument("--design_name", default="ariane133", help="the parent dir of design_name")
parser.add_argument("--cache_root", default="./cache", help="save path")
parser.add_argument("--unit", default=2000, help="unit defined in the begining of DEF")

# training parameters
parser.add_argument("--seed", type=int, default=42, metavar="N", help="random seed (default: 0)")
parser.add_argument("--gamma", type=float, default=0.95, metavar="G", help="discount factor (default: 0.9)")
parser.add_argument("--lr", type=float, default=2.5e-3)
parser.add_argument("--mini_batch", type=int, default=64)
parser.add_argument("--num_game_per_update", type=int, default=5)
parser.add_argument("--total_episodes", type=int, default=10000, help="number of episodes")

# result and log
parser.add_argument("--result_root", type=str, default="./result/", help="path for save result")
parser.add_argument("--log_root", type=str, default="./logs/", help="path for save result")
parser.add_argument("--update_threshold_ratio", type=float, default=0.97, help="update threshold ratio")

# run mode parameters
parser.add_argument("--cuda", type=str, default="", help="cuda device for set CUDA_VISIBLE_DEVICES")
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
parser.add_argument("--noorient", action="store_true", default=False, help="disable orient mode")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train_place", action="store_true", help="train place agent")
group.add_argument("--train_orient", action="store_true", help="train orient agent")
group.add_argument("--train_both", action="store_true", help="train both place and orient agent")

args = parser.parse_args()

set_random_seed(args.seed)
# set device to cpu or cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

benchmark = args.design_name
result_root = args.result_root
log_root = args.log_root

DesignReader = get_design_reader(args.design_name)
reader = DesignReader(**vars(args))
placedb = build_soft_macro_placedb(reader)

grid = 224
placed_num_macro = len(placedb.macro_info)
env = gym.make("orient_env-v0", placedb=placedb, grid=grid).unwrapped

print(f"seed = {args.seed}")
print(f"lr = {args.lr}")
print(f"placed_num_macro = {placed_num_macro}")

agent = OrientPPO(placed_num_macro, grid, args.mini_batch, args.lr, args.gamma, device)
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE

if args.train_place:
    agent.train_place_agent = True
    agent.train_orient_agent = False
elif args.train_orient:
    agent.train_place_agent = False
    agent.train_orient_agent = True
elif args.train_both:
    agent.train_place_agent = True
    agent.train_orient_agent = True


strftime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
RUN_ID = f"{placedb.design_name}_{strftime}_seed_{args.seed}_pnm_{placed_num_macro}_gamesperupdate_{args.num_game_per_update}_{'place' if args.noorient else 'orient'}"
if args.debug:
    RUN_ID = "DEBUG_" + RUN_ID
RESULT_PATH = pathlib.Path(result_root) / RUN_ID
LOG_PATH = pathlib.Path(log_root)
LOG_PATH.mkdir(parents=True, exist_ok=True)

FIGURE_PATH = RESULT_PATH / "figures"
MODEL_PATH = RESULT_PATH / "saved_model"
PLACE_PATH = RESULT_PATH / "placement"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLACE_PATH.mkdir(parents=True, exist_ok=True)

def main():

    running_reward = -float("inf")
    best_reward = running_reward
    last_record = 0

    trajectory = defaultdict(list)
    for i_epoch in range(args.total_episodes):
        last_record -= 1
        score = 0
        raw_score = 0
        start = time.time()
        state = env.reset()

        done = False
        while not done:
            orient_info, action_info, state_imgs = agent.select_action(state, noorient=args.noorient)
            orient, orient_log_prob, orient_value = orient_info
            action, action_log_prob, action_value = action_info
            macro_id, canvas, wire_img_8oc, pos_mask_8oc, wire_img_1oc, pos_mask_1oc = state_imgs

            next_state, reward, done, info = env.step(action, orient)
            assert next_state.shape == (3 + 17 * grid * grid,)
            score += reward
            raw_score += info["raw_reward"]
            state = next_state

            trajectory["macro_id"].append(macro_id)
            trajectory["canvas"].append(canvas)
            trajectory["wire_img_8oc"].append(wire_img_8oc)
            trajectory["pos_mask_8oc"].append(pos_mask_8oc)
            trajectory["wire_img_1oc"].append(wire_img_1oc)
            trajectory["pos_mask_1oc"].append(pos_mask_1oc)

            trajectory["orient"].append(orient)
            trajectory["action"].append(action)
            trajectory["reward"].append(np.float32(reward/2000))
            trajectory["o_log_prob"].append(np.float32(orient_log_prob))
            trajectory["a_log_prob"].append(np.float32(action_log_prob))
            trajectory["o_value"].append(np.float32(orient_value))
            trajectory["a_value"].append(np.float32(action_value))

            if done:
                end = time.time()
                print(f"Game Time: {end - start:.2f}s")
        
                if (i_epoch + 1) % args.num_game_per_update == 0:
                    cum_reward = 0.0
                    for index in reversed(range(len(trajectory["reward"]))):
                        reward = trajectory["reward"][index]
                        cum_reward = reward + args.gamma * cum_reward
                        trajectory["return"].append(np.float32(cum_reward))
                        trajectory["o_advantage"].append(np.float32(cum_reward) - trajectory["o_value"][index])
                        trajectory["a_advantage"].append(np.float32(cum_reward) - trajectory["a_value"][index])
                    trajectory["return"] = list(reversed(trajectory["return"]))
                    trajectory["o_advantage"] = list(reversed(trajectory["o_advantage"]))
                    trajectory["a_advantage"] = list(reversed(trajectory["a_advantage"]))

                    for key in trajectory:
                        values = trajectory[key]
                        if isinstance(values[0], np.ndarray):
                            # print(f"{key}: {len(values)}, {values[0].shape}, {np.stack(values).shape}")
                            trajectory[key] = np.stack(values)
                        else:
                            # print(f"{key}: {len(values)}, {np.array(values).shape}")
                            trajectory[key] = np.array(values)

                    agent.update(trajectory)
                    trajectory.clear()

                    print(f"Train Time: {time.time() - end:.2f}")

        

        if i_epoch == 0:
            running_reward = score
        running_reward = running_reward * 0.9 + score * 0.1
        print(f"score = {score:.3e}, raw_score = {raw_score:.3e}")

        if running_reward > best_reward * args.update_threshold_ratio or args.debug or last_record <= 0:
            best_reward = running_reward
            last_record = 100
            try:
                print("start try")
                # cost is the routing estimation based on the MST algorithm
                hpwl, cost = env.calc_hpwl_and_cost()
                print(f"hpwl = {hpwl:.3e}\tcost = {cost:.3e}")

                strftime_now = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
                save_flag = f"{strftime_now}_I[{i_epoch}]_S[{int(raw_score)}]_H[{hpwl:.3e}]_C[{cost:.3e}]"

                agent.save_model(MODEL_PATH, save_flag)
                fig_name = FIGURE_PATH / f"{save_flag}.png"
                env.plot(fig_name)
                print(f"save_figure: {fig_name}")
                pl_name = PLACE_PATH / f"{save_flag}.pl"
                env.save_pl_file(pl_name)
            except:
                assert False

        if i_epoch % 1 == 0:
            print(f"Epoch {i_epoch}, Moving average score is: {running_reward:.3e} Best reward: {best_reward:.3e}")

        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break


if __name__ == "__main__":
    main()
