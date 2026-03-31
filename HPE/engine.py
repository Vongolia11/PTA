import os
import argparse
import torch

from torch.utils.data import DataLoader
from logger import get_logger
from misc import extant_file

logger = get_logger()

class Engine:
    def __init__(self, custom_parser=None):
        logger.info("PyTorch Version {}".format(torch.__version__))

        self.devices = None

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        # 获取 CUDA_VISIBLE_DEVICES 中的 GPU 列表
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.devices = [i for i in range(len(gpus.split(','))) if gpus]

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
    def data_parallel(self, model):
        return torch.nn.DataParallel(model)

    def get_train_loader(self, train_dataset, batch_size=None, collate_fn=None):
        if batch_size is None:
            batch_size = self.args.batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            collate_fn=collate_fn
        )

        return train_loader

    def get_test_loader(self, test_dataset, batch_size=1):
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

        return test_loader

    def all_reduce_tensor(self, tensor, norm=True):
        return torch.mean(tensor)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning("Exception during Engine, aborting.")
            return False
