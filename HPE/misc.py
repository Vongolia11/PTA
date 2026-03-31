import argparse
import os
import random
from typing import List


def random_num_select(inputs:List):
    modality_num = len(inputs)
    select_input = random.sample(inputs, random.randint(1, modality_num))
    other_input = set(inputs) - set(select_input)
    
    return select_input, other_input

def all_num_select(inputs: List[str]):
    return inputs, []
def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

