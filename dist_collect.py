import os
import gym
import time
import reverb
import argparse
import tensorflow as tf
from loguru import logger

import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import OrientPPO
from reverb_util import tf_to_torch
from util import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--gamma", type=float, default=0.98)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reverb_ip", type=str, default="localhost")
parser.add_argument("--reverb_port", type=int, default=12888)
parser.add_argument("--cuda", type=str, default="")
args = parser.parse_args()

if args.seed is None:
    args.seed = int(time.time())
    logger.info(f"Set random seed to {args.seed}")
set_random_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

PROJECT_ROOT = args.project_root
data_root = os.path.join(PROJECT_ROOT, "benchmark")
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, args.design_name, cache_root)
placedb = build_soft_macro_placedb(reader, cache_root=cache_root)
env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
agent = OrientPPO(len(placedb.macro_info), grid=224, num_game_per_update=5, batch_size=128, lr=1e-5, gamma=0.98, device='cuda')
agent.CANVAS_SLICE = env.CANVAS_SLICE
agent.WIRE_SLICE = env.WIRE_SLICE
agent.POS_SLICE = env.POS_SLICE
agent.FEATURE_SLICE = env.FEATURE_SLICE

REVERB_ADDR = f"{args.reverb_ip}:{args.reverb_port}"
MODEL_INFO = reverb.TimestepDataset.from_table_signature(
    server_address=REVERB_ADDR,
    table="model_info",
    max_in_flight_samples_per_worker=1
)

def collect():
    while True:
        t0 = time.time()
        model = next(iter(MODEL_INFO.take(1)))
        model_data = model.data
        model_data = tf.nest.map_structure(tf_to_torch, model_data)
        model_id = model_data.pop('model_id').numpy().item()
        orient_actor = model_data['orient_actor']
        agent.orient_actor_net.load_state_dict(orient_actor)
        place_actor = model_data['place_actor']
        agent.place_actor_net.load_state_dict(place_actor)
        t1 = time.time()

        with reverb.Client(REVERB_ADDR).trajectory_writer(num_keep_alive_refs=200) as writer:
            state = env.reset()
            done = False
            total_reward = 0
            trajectory = []
            while not done:
                orient, action, orient_log_prob, action_log_prob = agent.select_action(state)
                next_state, reward, done, info = env.step(action, orient)
                total_reward += reward
                trajectory.append({
                    'state': state,
                    'orient': orient,
                    'action': action,
                    'o_log_prob': orient_log_prob,
                    'a_log_prob': action_log_prob,
                    'reward': float(reward)/2000,
                    'next_state': next_state,
                    'done': done,
                    'model_id': model_id,
                })
                state = next_state
            t2 = time.time()

            cum_reward = 0
            for step_log in reversed(trajectory):
                reward = step_log['reward']
                cum_reward = reward + args.gamma * cum_reward
                step_log['return'] = cum_reward
            for step_log in trajectory:
                writer.append(step_log)
            writer.create_item('experience', model_id, 
                            trajectory= {
                                'state': writer.history['state'][:],
                                'orient': writer.history['orient'][:],
                                'action': writer.history['action'][:],
                                'o_log_prob': writer.history['o_log_prob'][:],
                                'a_log_prob': writer.history['a_log_prob'][:],
                                'reward': writer.history['reward'][:],
                                'next_state': writer.history['next_state'][:],
                                'done': writer.history['done'][:],
                                'return': writer.history['return'][:],
                                'model_id': writer.history['model_id'][-1]
                            })
            writer.flush()
            t3 = time.time()
            logger.info(f"{model_id=}, {total_reward=:.3e}, total={t3-t0:.3f}, read_model={t1-t0:.3f}, gen_sample={t2-t1:.3f}, push_sample={t3-t2:.3f}")


if __name__ == "__main__":
    collect()