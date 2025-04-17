import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--gpus", type=int, default=4)
parser.add_argument("--noport", action="store_true", default=False, help="PLACEENV_IGNORE_PORT=1")


parser.add_argument("--design_name", type=str, default="ariane133")
parser.add_argument("--project_root", type=str, default=".")

parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--reverb_ip", type=str, default="localhost")
parser.add_argument("--reverb_port", type=int, default=12888)

args = parser.parse_args()

for i in range(args.jobs):
    command = f"python dist_collect.py --design_name {args.design_name} --reverb_ip {args.reverb_ip} --cuda {i % args.gpus} > collect_{args.design_name}_{i}.log 2>&1 &"
    if args.ignore_port:
        command = "PLACEENV_IGNORE_PORT=1 " + command
    os.system(command)