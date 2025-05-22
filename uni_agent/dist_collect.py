import os
import sys
import gym
import time
import reverb
import argparse
import numpy as np
import tensorflow as tf
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import place_env
from placedb import LefDefReader, build_soft_macro_placedb
from model import UniPPO
from reverb_util import tf_to_torch
from util import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--noorient", action="store_true", default=False, help="noorient")

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reverb_ip", type=str, default="localhost")
parser.add_argument("--reverb_port", type=int, default=12888)
parser.add_argument("--cuda", type=str, default="")
args = parser.parse_args()

if args.seed is None:
    args.seed = int(time.time())
    logger.info(f"Set random seed to {args.seed}")
else:
    logger.info(f"Set seed to {args.seed}")
set_random_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

PROJECT_ROOT = args.project_root
data_root = os.path.join(PROJECT_ROOT, "benchmark")
cache_root = os.path.join(PROJECT_ROOT, "cache")
reader = LefDefReader(data_root, args.design_name, cache_root)
placedb = build_soft_macro_placedb(reader, cache_root=cache_root)
env = gym.make("orient_env-v0", placedb=placedb, grid=224).unwrapped
agent = UniPPO(len(placedb.macro_info), grid=224, batch_size=128, lr=1e-5, device='cuda')
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
        model_variables = model.data

        model_variables = tf.nest.map_structure(tf_to_torch, model_variables)
        model_id = model_variables.pop('model_id').numpy().item()
        agent.uni_actor_net.load_state_dict(model_variables['uni_actor'])
        agent.uni_critic_net.load_state_dict(model_variables['uni_critic'])

        t1 = time.time()

        with reverb.Client(REVERB_ADDR).trajectory_writer(num_keep_alive_refs=200) as writer:
            state = env.reset()
            done = False
            total_reward = 0
            trajectory = []
            while not done:
                action_info, state_info = agent.select_action(state, args.noorient)
                action, log_prob, value = action_info
                state_, mask, macro_id = state_info
                
                orient = action // 224 // 224
                position = action % (224 * 224)
                next_state, reward, done, _ = env.step(position, orient)
                total_reward += reward

                trajectory.append({
                    'macro_id': macro_id,
                    'state': state_,
                    'mask': mask,

                    'action': action,
                    'reward': np.float32(reward/2000),
                    'log_prob': np.float32(log_prob),
                    'value': np.float32(value),
                    'model_id': model_id,
                })
                state = next_state
            t2 = time.time()

            cum_reward = 0
            for step_log in reversed(trajectory):
                reward = step_log['reward']
                cum_reward = reward + args.gamma * cum_reward
                step_log['return'] = np.float32(cum_reward)
                step_log['advantage'] = np.float32(cum_reward) - step_log['value']
            
            for step_log in trajectory:
                writer.append(step_log)
            writer.create_item('experience', model_id, 
                            trajectory= {
                                'macro_id': writer.history['macro_id'][:],
                                'state': writer.history['state'][:],
                                'mask': writer.history['mask'][:],

                                'action': writer.history['action'][:],
                                'reward': writer.history['reward'][:],
                                'log_prob': writer.history['log_prob'][:],
                                'value': writer.history['value'][:],
                                'return': writer.history['return'][:],
                                'advantage': writer.history['advantage'][:],
                                'model_id': writer.history['model_id'][-1]
                            })
            writer.flush()
            t3 = time.time()
            logger.info(f"{model_id=}, {total_reward=:.3e}, total={t3-t0:.3f}, read_model={t1-t0:.3f}, gen_sample={t2-t1:.3f}, push_sample={t3-t2:.3f}")


if __name__ == "__main__":
    collect()