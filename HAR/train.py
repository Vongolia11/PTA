import argparse
import sys

from torch.utils.hipify.hipify_python import str2bool

from HAR_Task import HAR_Task
from Extractor import *

def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="HAR start")
    parser.add_argument("--data_dir", type=str, default='./Data/XRF55_Dataset')
    parser.add_argument("--reload_path", type=str, default='./checkpoint/example/')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--snapshot_dir", type=str, default='./checkpoint/example/')

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=30000)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=212)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=34)

    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--mode", type=str, default='random')

    return parser

def main():
    sys.path.append('HAR')
    parser = get_arguments()
    print(parser)
    har_task = HAR_Task()
    har_task.train(parser)


if __name__ == "__main__":
    main()