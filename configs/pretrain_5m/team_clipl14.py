exp_dir = 'snapshots/opensrc_team_clip_l14'
eval_only = False
start_epoch = 0
IMG_SIZE = 224

#####################################################################
############################# model #############################
#####################################################################
import torch
import torch.nn as nn
from models.backbone.bert import BertWrapper
from models.backbone.clip.clip import load
from models.team import TEAM
def get_model(pretrained=None):
    text_model = BertWrapper(hf_bert_dir='pretrained/bert_base_uncased', use_cls_token=True, use_gradient_ckpt=False, feat_dim=768, token_dim=1024)
    text_model.bert.cls = None
    NEG_NUM = 1
    image_model, _ = load('pretrained/ViT-L-14.pt', use_gc=True, large_model=True)
    model = TEAM(
        text_model=text_model,
        image_model=image_model,
        pretrained=pretrained,
        neg_num=NEG_NUM,
        team_block_num=6,
        mask_noise=True,
        retrieve_mlp_ratio=2,
        retrieve_use_dropout=True,
        token_dim=1024)
    return model

#####################################################################
############################# optimizer #############################
#####################################################################
def get_params_group(model, lr_default, logger=None):
    orig_params = []
    low_lr_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if ('image_model' in name) and 'tensor_fc' not in name:
            low_lr_params.append(param)
            if logger is not None:
                logger.info('{} has low lr (1/10)'.format(name))
        else:
            orig_params.append(param)
            if logger is not None:
                logger.info('{} has orig lr'.format(name))
    return [
        {'params': orig_params, 'lr': lr_default},
        {'params': low_lr_params, 'lr': lr_default / 10.0}]

import apex.optimizers.fused_lamb as fused_lamb
def get_optimizer(model, logger=None):
    learning_rate_default = 1e-3
    betas = [0.9, 0.999]
    weight_decay = 0.01
    optimizer = fused_lamb.FusedLAMB(
        get_params_group(model, learning_rate_default, logger),
        lr=learning_rate_default,
        betas=betas,
        weight_decay=weight_decay)
    return optimizer

#####################################################################
############################# scheduler #############################
#####################################################################
def get_scheduler(optimizer):
    lr_lmbda = lambda epoch: 0.1 ** (epoch // 10)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lmbda)

#####################################################################
############################# dataset ###############################
#####################################################################
import pickle
import io
import tempfile
from tokenizers import BertWordPieceTokenizer
from datasets.utils import collate_fn30
from datasets.en_pretrain import get_pretrain_dataloaders
from datasets.coco import get_coco2014_eval_dataloaders
def get_tokenizer():
    text_tokenizer = BertWordPieceTokenizer('pretrained/bert_base_uncased/bert-base-uncased-vocab.txt', lowercase=True)
    text_tokenizer.enable_truncation(max_length=30)
    return text_tokenizer
def get_train_dataloader(epoch):
    TRAIN_BATCH_SIZE = 240
    text_tokenizer = get_tokenizer()
    return get_pretrain_dataloaders(TRAIN_BATCH_SIZE, epoch, text_tokenizer, img_reso=IMG_SIZE, collate_fn=collate_fn30, num_workers=16)
def get_val_dataloaer():
    EVAL_BATCH_SIZE = 480
    text_tokenizer = get_tokenizer()
    return get_coco2014_eval_dataloaders(EVAL_BATCH_SIZE, text_tokenizer, img_reso=IMG_SIZE, collate_fn=collate_fn30, pretrain=True)
