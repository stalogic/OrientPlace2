import argparse
import os
import pathlib
import time
from collections import namedtuple

import gym
import torch

import place_env
from model import PlacePPO, OrientPPO
from placedb import PlaceDB, LefDefReader, build_soft_macro_placedb
from util import set_random_seed

Transition = namedtuple(
    "Transition",
    [
        "state",
        "orient",
        "action",
        "reward",
        "o_log_prob",
        "a_log_prob",
        "next_state",
        "reward_intrinsic",
    ],
)


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
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--total_episodes", type=int, default=10000, help="number of episodes")

# result and log
parser.add_argument("--result_root", type=str, default="./result/", help="path for save result")
parser.add_argument("--log_root", type=str, default="./logs/", help="path for save result")
parser.add_argument("--update_threshold_ratio", type=float, default=0.975, help="update threshold ratio")

# run mode parameters
parser.add_argument("--cuda", type=str, default="", help="cuda device for set CUDA_VISIBLE_DEVICES")
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
parser.add_argument("--noorient", action="store_true", default=False, help="disable orient mode")
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

lefdef_reader = LefDefReader(**vars(args))
placedb = PlaceDB(
    design_name=lefdef_reader.design_name,
    place_net_dict=lefdef_reader.place_net_dict,
    place_instance_dict=lefdef_reader.place_instance_dict,
    place_pin_dict=lefdef_reader.place_pin_dict,
    lef_dict=lefdef_reader.lef_dict,
    die_area=lefdef_reader.die_area,
)

# placedb = convert_to_soft_macro_placedb(placedb=placedb, parser=lefdef_reader)

grid = 224
placed_num_macro = len(placedb.macro_info)
env = gym.make("orient_env-v0", placedb=placedb, grid=grid).unwrapped

print(f"seed = {args.seed}")
print(f"lr = {args.lr}")
print(f"placed_num_macro = {placed_num_macro}")


def main():

    strftime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    RUN_ID = f"{benchmark}_{strftime}_seed_{args.seed}_pnm_{placed_num_macro}_{'place' if args.noorient else 'orient'}"
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

    log_file_name = LOG_PATH / f"{RUN_ID}.csv"
    fwrite = open(log_file_name, "w")

    if args.noorient:
        agent = PlacePPO(placed_num_macro, grid, args.batch_size, args.lr, args.gamma, device)
    else:
        agent = OrientPPO(placed_num_macro, grid, args.batch_size, args.lr, args.gamma, device)
    agent.CANVAS_SLICE = env.CANVAS_SLICE
    agent.WIRE_SLICE = env.WIRE_SLICE
    agent.POS_SLICE = env.POS_SLICE
    agent.FEATURE_SLICE = env.FEATURE_SLICE

    running_reward = -float("inf")
    best_reward = running_reward

    for i_epoch in range(args.total_episodes):
        score = 0
        raw_score = 0
        start = time.time()
        state = env.reset()

        done = False
        while done is False:
            state_tmp = state.copy()
            orient, action, orient_log_prob, action_log_prob = agent.select_action(state)

            next_state, reward, done, info = env.step(action, orient)
            assert next_state.shape == (3 + 17 * grid * grid,)
            reward_intrinsic = 0
            trans = Transition(
                state_tmp,
                orient,
                action,
                reward / 200.0,
                orient_log_prob,
                action_log_prob,
                next_state,
                reward_intrinsic,
            )
            if agent.store_transition(trans):
                assert done == True
                agent.update()
            score += reward
            raw_score += info["raw_reward"]
            state = next_state
        end = time.time()

        print(f"Game Time: {end - start:.2f}s")

        if i_epoch == 0:
            running_reward = score
        running_reward = running_reward * 0.9 + score * 0.1
        print(f"score = {score:.3e}, raw_score = {raw_score:.3e}")

        if running_reward > best_reward * args.update_threshold_ratio or args.debug:
            best_reward = running_reward
            if i_epoch > 10 or args.debug:
                try:
                    print("start try")
                    # cost is the routing estimation based on the MST algorithm
                    hpwl, cost = env.calc_hpwl_and_cost()
                    print(f"hpwl = {hpwl:.3e}\tcost = {cost:.3e}")

                    strftime_now = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
                    save_flag = f"{strftime_now}_I[{i_epoch}]_S[{int(raw_score)}]_H[{hpwl:.3e}]_C[{cost:.3e}]"

                    agent.save_param(MODEL_PATH / save_flag)
                    fig_name = FIGURE_PATH / f"{save_flag}.png"
                    env.save_flyline(fig_name)
                    print(f"save_figure: {fig_name}")
                    pl_name = PLACE_PATH / f"{save_flag}.pl"
                    env.save_pl_file(pl_name)
                except:
                    assert False

        if i_epoch % 1 == 0:
            print(f"Epoch {i_epoch}, Moving average score is: {running_reward:.3e} Best reward: {best_reward:.3e}")
            fwrite.write(f"{i_epoch},{score:.3e},{running_reward:.3e},{best_reward:.3e},{agent.training_step}\n")
            fwrite.flush()
        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break


if __name__ == "__main__":
    main()
