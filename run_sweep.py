import argparse
from datetime import datetime

import wandb
from configs.utils import str2bool
from experiments.evaluate import add_const_param, add_name
from experiments.utils import read_yaml

TIME_STR = '{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now())
DATE_str = '{:%Y_%m_%d}'.format(datetime.now())

PATH_TO_CONFIGS = "./configs/sweeps/"


def parse_default_args():
    parser = argparse.ArgumentParser(description='GraphInterventionNetworks')
    parser.add_argument('--name', type=str, default=TIME_STR)
    parser.add_argument('--task', type=str, default='sw',
                        choices=['sw', 'tcga'])
    parser.add_argument('--model', type=str, default='gin',
                        choices=['gin', 'gnn', 'graphite', 'cat'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_sweeps', type=int, default=10)
    parser.add_argument('--bias', type=float, default=10.)
    parser.add_argument('--ablation', type=str2bool, default=False,
                        help='Changes wandb project name.')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='Alpha')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_default_args()
    project_name = f"GIN_{DATE_str}-{args.task}-ABL" if args.ablation else f"GIN_{DATE_str}-{args.task}"
    wandb.init(project=project_name)
    wandb.config.update(args)
    yaml_path = PATH_TO_CONFIGS + f"{args.task}/{args.model}.yaml"
    sweep_config = read_yaml(path=yaml_path)
    sweep_run_name = f'{args.model}-{args.seed}-{args.bias}'
    add_name(sweep_run_name, sweep_config)
    add_const_param('cuda', args.cuda, sweep_config)
    add_const_param('ablation', args.ablation, sweep_config)
    add_const_param('bias', args.bias, sweep_config)
    add_const_param('seed', args.seed, sweep_config)
    add_const_param('model', args.model, sweep_config)
    add_const_param('task', args.task, sweep_config)

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id, count=args.num_sweeps)
