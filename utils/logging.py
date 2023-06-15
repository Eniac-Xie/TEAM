import logging 
import os
import sys
import time

import torch.distributed as dist

class MultiModalLogging(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    def add_file(self, log_dir):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            file_handler = logging.FileHandler('{}/{}.log'.format(log_dir, timestamp_str))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

    def add_std(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(self.formatter)
            self.logger.addHandler(ch)

    def get(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.ERROR)
        return self.logger


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val_list = []
        self.avg = 0

    def update(self, val):
        self.val_list.append(val)
        if len(self.val_list) > 50:
            self.val_list.pop(0)
        self.avg = sum(self.val_list) / len(self.val_list)
