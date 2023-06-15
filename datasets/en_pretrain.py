import os
import time
from utils.logging import MultiModalLogging

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.coco import COCODataset
from datasets.utils import collate_fn30

logging = MultiModalLogging()
logger = logging.get()

class EnPretrainDataset(Dataset):
    def __init__(self, text_tokenizer, img_reso=224, mask_text=True):
        self.cc3m = COCODataset('data/cc3m/cc_albef_qy.json', 'data/cc3m', 'train', text_tokenizer, max_words=30, img_reso=img_reso, pretrain=True)
        
        self.vg = COCODataset('data/VG/vg_pretrain.json', 'data', 'train', text_tokenizer, max_words=30, img_reso=img_reso, pretrain=True)

        self.coco_trainval = COCODataset('data/coco2014/annotations/albef_coco_pretrain.json', 'data/coco2014', 'train', text_tokenizer, max_words=30, img_reso=img_reso, pretrain=True)

        self.sbu = COCODataset('data/SBU/sbucaptions/sbu_albef.json', 'data/SBU/sbucaptions', 'train', text_tokenizer, max_words=30, img_reso=img_reso, pretrain=True)

        self.cc3m_len = len(self.cc3m)
        self.vg_len = len(self.vg)
        self.coco_trainval_len = len(self.coco_trainval)
        self.sbu_len = len(self.sbu)

        t1 = time.time()
        self.all_oss_path2img_idx = self.get_all_oss_path2img_idx()
        t2 = time.time()
        logger.info('get_all_oss_path2img_idx takes {:.2f} sec'.format(t2-t1))
    
    def get_oss_path2img_idx(self, dataset, oss_path2img_idx):
        for index in range(len(dataset)):
            oss_path = os.path.join(dataset.image_root, dataset.ann[index]['image'])
            if oss_path not in oss_path2img_idx:
                oss_path2img_idx[oss_path] = len(oss_path2img_idx)
    
    def get_all_oss_path2img_idx(self):
        all_oss_path2img_idx = {}
        self.get_oss_path2img_idx(self.cc3m, all_oss_path2img_idx)
        self.get_oss_path2img_idx(self.vg, all_oss_path2img_idx)
        self.get_oss_path2img_idx(self.coco_trainval, all_oss_path2img_idx)
        self.get_oss_path2img_idx(self.sbu, all_oss_path2img_idx)
        return all_oss_path2img_idx

    def __len__(self):
        return self.cc3m_len + self.vg_len + self.coco_trainval_len + self.sbu_len
    
    def post_process(self, mm_data):
        misc = mm_data.misc
        oss_path = misc.rsplit(':', 1)[0]
        try:
            img_idx = self.all_oss_path2img_idx[oss_path]
        except Exception as err:
            print('find img_idx failed')
            img_idx = -1
        mm_data.misc = img_idx
        return mm_data

    def __getitem__(self, index):
        if index < self.cc3m_len:
            mm_data = self.cc3m.__getitem__(index)
        elif index < self.cc3m_len + self.vg_len:
            mm_data = self.vg.__getitem__(index - self.cc3m_len)
        elif index < self.cc3m_len + self.vg_len + self.coco_trainval_len:
            mm_data = self.coco_trainval.__getitem__(index - self.cc3m_len - self.vg_len)
        elif index < self.cc3m_len + self.vg_len + self.coco_trainval_len + self.sbu_len:
            mm_data = self.sbu.__getitem__(index - self.cc3m_len - self.vg_len - self.coco_trainval_len)
        else:
            raise ValueError
        return self.post_process(mm_data)


def get_pretrain_dataloaders(batch_size, epoch, text_tokenizer, img_reso=224, collate_fn=collate_fn30, num_workers=8):
    for _ in range(1):
        train_dataset = EnPretrainDataset(text_tokenizer, img_reso=img_reso)
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        train_sampler.set_epoch(epoch)
        
        train_params = {
                        'pin_memory': True,
                        'collate_fn': collate_fn,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'drop_last': True,
                        'sampler': train_sampler,
                        'num_workers': num_workers}

        train_loader = DataLoader(train_dataset, **train_params)
        yield (train_loader, 'pretrain')


if __name__ == '__main__':
    import pickle
    import io
    import tempfile
    from tokenizers import BertWordPieceTokenizer
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', world_size=1, rank=0)
    TRAIN_BATCH_SIZE = 240
    EVAL_BATCH_SIZE = 480
    text_tokenizer = BertWordPieceTokenizer('pretrained/bert_base_uncased/bert-base-uncased-vocab.txt', lowercase=True)
    text_tokenizer.enable_truncation(max_length=30)
    for epoch in range(30):
        dataloader_list = get_pretrain_dataloaders(TRAIN_BATCH_SIZE, epoch, text_tokenizer)
        for dataloader, _ in dataloader_list:
            for data in dataloader:
                import pdb;pdb.set_trace()
                break
