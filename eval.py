import os
import sys
import time
import argparse
from importlib import import_module
import torch
import torch.nn.functional as F
import numpy as np
from utils.eval_utils import evaluation_retrieve, padding_tokens_30
from utils.logging import MultiModalLogging
from datasets.coco_retrieve import get_coco2014_retrieve_dataloaders

logging = MultiModalLogging()
logging.add_std()
logger = logging.get()


parser = argparse.ArgumentParser(description='TEAM Evaluation')
parser.add_argument('config', type=str, help='path to config file')
args = parser.parse_args()


config_dir = os.path.dirname(args.config)
config_name = os.path.basename(args.config).rsplit('.', 1)[0]
sys.path.insert(0, config_dir)
config = import_module(config_name)


model = config.get_model('pretrained/pretrained_14m_clip_large.pth')

get_dataloader_func = get_coco2014_retrieve_dataloaders
re_val_dataloader_list = get_dataloader_func(batch_size=64, img_reso=config.IMG_SIZE)

text_tokenizer = config.get_tokenizer()
for re_val_dataloader, _ in re_val_dataloader_list:
    evaluation_retrieve(model, re_val_dataloader, text_tokenizer, device=torch.device('cuda'), two_stage=True, padding_tokens=padding_tokens_30)
