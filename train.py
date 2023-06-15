import warnings
warnings.filterwarnings("ignore")
import io
import os
import sys
import argparse
import random
import time
import logging
from importlib import import_module
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import get_world_size, get_rank

parser = argparse.ArgumentParser(description='TEAM Training')
parser.add_argument('config', type=str, help='path to config file')
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

from utils.logging import MultiModalLogging
from utils.logging import AverageMeter


def train_epoch(ddp_model, optimizer, train_loader, epoch, dist_info, logger, amp_scaler, config, node_group, use_amp):
    ddp_model.train()

    data_time_metric = AverageMeter('Data Time')
    forward_time_metric = AverageMeter('Forward Time')
    backward_time_metric = AverageMeter('Backward Time')

    torch.cuda.synchronize()
    t1 = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        torch.cuda.synchronize()
        data_time = time.time() - t1
        t1 = time.time()

        log_info = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'all_batch_cnt': len(train_loader)
        }
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            losses = ddp_model.forward(batch_data, dist_info, batch_idx % 10 == 0, log_info, phase='train', node_group=node_group)

        torch.cuda.synchronize()
        forward_time = time.time() - t1
        t1 = time.time()

        amp_scaler.scale(losses).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        torch.cuda.synchronize()
        backward_time = time.time() - t1

        data_time_metric.update(data_time)
        forward_time_metric.update(forward_time)
        backward_time_metric.update(backward_time)

        if batch_idx % 10 == 0:
            logger.info('Data Time: {:.3f}, Forward Time: {:.3f}, Backward Time: {:.3f}, amp: {:.5f}'.format(
                data_time_metric.avg, forward_time_metric.avg, backward_time_metric.avg, amp_scaler.get_scale()))

        if batch_idx % 1000 == 0 and batch_idx != 0:
            global_rank = dist_info['global_rank']
            if global_rank == 0:
                logger.info('saving models')
                torch.save(ddp_model.state_dict(), '{}/epoch{}_{}_params.pth'.format(config.exp_dir, epoch, batch_idx))

        t1 = time.time()


def eval_epoch(ddp_model, eval_loader, epoch, dist_info, logger, exp_dir):
    ddp_model.eval()
    
    with torch.no_grad():
        tp_cnt, all_cnt = 0.0, 0.0
        device = torch.device('cuda:{}'.format(dist_info['local_rank']))
        for batch_idx, batch_data in enumerate(eval_loader):
            with torch.cuda.amp.autocast():
                match_scores = ddp_model.forward(batch_data, dist_info, phase='eval')
            text_cnt, img_cnt = match_scores.shape
            assert text_cnt == img_cnt
            _, match_ids = torch.max(match_scores, dim=1)
            match_ids = match_ids.int()
            gt_ids = torch.tensor(range(0, text_cnt)).to(device, non_blocking=True).int()
            error_cnt = torch.nonzero(match_ids - gt_ids)
            all_cnt += text_cnt
            tp_cnt += (text_cnt - 1.0 * error_cnt.numel())
            logger.info('batch-level retrieval accuracy (NOT the final metric): {:.3f}'.format(tp_cnt / all_cnt))


def worker(local_rank, dist_world_size, global_rank): 
    dist.init_process_group(backend='nccl', world_size=dist_world_size, rank=global_rank)

    config_dir = os.path.dirname(args.config)
    config_name = os.path.basename(args.config).rsplit('.', 1)[0]
    sys.path.insert(0, config_dir)
    config = import_module(config_name)

    mm_logging = MultiModalLogging()
    logger = mm_logging.get()
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)
    mm_logging.add_std()
    mm_logging.add_file(config.exp_dir)
    
    logger.info('exp_dir: {}'.format(config.exp_dir))
    logger.info('GPU info: {}'.format(torch.cuda.get_device_name(0)))
    logger.info('local_rank: {}, global_rank: {}, get_rank(): {}, dist_world_size: {}, get_world_size(): {}'.format(
        local_rank, global_rank, get_rank(), dist_world_size, get_world_size()))

    if hasattr(config, 'use_node_group') and config.use_node_group:
        raise NotImplementedError
    else:
        node_group = None
 
    if hasattr(config, 'use_amp') and not config.use_amp:
        use_amp = False
    else:
        use_amp = True

    assert global_rank == get_rank()
    assert dist_world_size == get_world_size()
    dist_info = {
        'local_rank': local_rank,
        'global_rank': global_rank,
        'dist_world_size': dist_world_size
    }

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:{}".format(local_rank))

    # init model here
    model = config.get_model()
    logger.info('model nparams: {}'.format(sum(p.numel() for p in model.parameters())))
    model.to(device)

    # init tokenizer
    text_tokenizer = config.get_tokenizer()

    # init ddp
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = config.get_optimizer(ddp_model, logger)
    scheduler = config.get_scheduler(optimizer)

    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if hasattr(config, 'resume'):
        resume_params = torch.load('{}_params.pth'.format(config.resume), 'cpu')
        ddp_model.load_state_dict(resume_params)
        logger.info('load params from {}'.format('{}_params.pth'.format(config.resume)))
        resume_opts = torch.load('{}_opt.pth'.format(config.resume), 'cpu')
        optimizer.load_state_dict(resume_opts)
        logger.info('load opts from {}'.format('{}_opt.pth'.format(config.resume)))
        resume_scaler = torch.load('{}_scaler.pth'.format(config.resume), 'cpu')
        amp_scaler.load_state_dict(resume_scaler)
        logger.info('load scaler from {}'.format('{}_scaler.pth'.format(config.resume)))
        config.start_epoch = int(config.resume.split('/')[-1].replace('epoch', '')) + 1

    if hasattr(config, 'nepoch'):
        EPOCH = config.nepoch
    else:
        EPOCH = 30
        
    for epoch in range(EPOCH):
        if epoch < config.start_epoch:
            logger.info('skip epoch {} of lr: {}'.format(epoch, scheduler.get_last_lr()))
            scheduler.step()
            continue
        
        if hasattr(config, 'skip_eval') and config.skip_eval:
            pass
        else:
            if global_rank == 0:
                all_eval_loaders = config.get_val_dataloaer()
                for eval_loader, eval_name in all_eval_loaders:
                    eval_epoch(ddp_model, eval_loader, epoch, dist_info, logger, config.exp_dir)
                if config.eval_only:
                    return
        
        all_train_loaders = config.get_train_dataloader(epoch)

        logger.info('epoch {} training starts, lr: {}'.format(epoch, scheduler.get_last_lr()))

        for train_loader, train_name in all_train_loaders:
            train_epoch(ddp_model, optimizer, train_loader, epoch, dist_info, logger, amp_scaler, config, node_group, use_amp)

        if global_rank == 0:
            logger.info('saving models')
            torch.save(ddp_model.state_dict(), '{}/epoch{}_params.pth'.format(config.exp_dir, epoch))
            torch.save(amp_scaler.state_dict(), '{}/epoch{}_scaler.pth'.format(config.exp_dir, epoch))
            torch.save(optimizer.state_dict(), '{}/epoch{}_opt.pth'.format(config.exp_dir, epoch))

        scheduler.step()


if __name__ == '__main__':
    print(os.environ["WORLD_SIZE"], os.environ["RANK"])
    worker(
        local_rank=int(os.environ['LOCAL_RANK']),
        dist_world_size=int(os.environ["WORLD_SIZE"]),
        global_rank=int(os.environ["RANK"])
    )
