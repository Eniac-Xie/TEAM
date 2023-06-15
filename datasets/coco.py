import json
import io
import os
import random
import tempfile
import re
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, \
        RandomHorizontalFlip, RandomResizedCrop
        
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from datasets.utils import MMData, collate_fn30

from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x   


class COCODataset(Dataset):
    def __init__(self,
                ann_file,
                image_root,
                phase,
                tokenizer,
                max_words=30,
                img_reso=224,
                pretrain=False,
                epoch=None,
                text_processor=pre_caption):
        assert phase in ('train', 'val')
        self.phase = phase
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.pretrain = pretrain

        self.ann = json.load(open(ann_file,'r'))

        if self.phase == 'val':
            self.transform = [
                    Compose(
                        [
                            Resize((img_reso, img_reso), interpolation=Image.BICUBIC),
                            ToTensor(),
                            Normalize(
                                (0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711))
                        ])
                ]
        else:
            from datasets.randaugment import RandomAugment
            self.transform = [
                    Compose([                        
                        transforms.RandomResizedCrop(img_reso, scale=(0.2 if pretrain else 0.5, 1.0), interpolation=Image.BICUBIC),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))
                    ])
            ]

        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(text_processor(caption, self.max_words) if text_processor is not None else caption)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
        
        if not self.pretrain:
            t1 = time.time()
            self.get_oss_path2img_idx()
            t2 = time.time()
            logger.info('get_oss_path2img_idx takes {} sec'.format(t2-t1))
    
    def get_oss_path2img_idx(self, ):
        self.oss_path2img_idx = {}
        for index in range(len(self.image)):
            oss_path = os.path.join(self.image_root, self.ann[index]['image'])
            if oss_path not in self.oss_path2img_idx:
                self.oss_path2img_idx[oss_path] = len(self.oss_path2img_idx)

    def post_process(self, mm_data):
        misc = mm_data.misc
        oss_path = misc.rsplit(':', 1)[0]
        try:
            img_idx = self.oss_path2img_idx[oss_path]
        except Exception as err:
            print('find img_idx failed')
            img_idx = -1
        mm_data.misc = img_idx
        return mm_data

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        for _ in range(10):
            try:
                oss_path = os.path.join(self.image_root, self.ann[index]['image'])
                image_pil = Image.open(oss_path).convert('RGB')
            except Exception as err:
                image_pil = None
        
            if image_pil is not None:
                break
            else:
                index = random.choice(range(len(self.image)))

        image = random.choice(self.transform)(image_pil)

        if self.epoch is None:
            text_str = self.text[random.choice(self.img2txt[index])]
        else:
            text_idx = self.img2txt[index][self.epoch % len(self.img2txt[index])]
            text_str = self.text[text_idx]
        text = self.tokenizer.encode(text_str)
        
        token_ids, token_attention_mask = text.ids, text.attention_mask

        data = MMData(
            torch.tensor(token_ids),
            torch.tensor(token_attention_mask),
            image,
            oss_path + ':' + text_str.replace(':', '_'),
            None,
            None,
            None,
            None)
        if not self.pretrain:
            data = self.post_process(data)
        return data


def get_coco2014_train_dataloaders(batch_size, epoch, text_tokenizer, img_reso=224, collate_fn=collate_fn30):
    train_list = [
            'data/coco2014/annotations/albef_coco_train_aggre.json'
    ]
    for train_file in train_list:
        train_dataset = COCODataset(train_file, 'data/coco2014', 'train', text_tokenizer, img_reso=img_reso)
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        train_sampler.set_epoch(epoch)

        num_workers = 4 if img_reso > 320 else 16
        train_params = {
                        'pin_memory': True,
                        'collate_fn': collate_fn,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'drop_last': True,
                        'sampler': train_sampler,
                        'num_workers': num_workers}

        train_loader = DataLoader(train_dataset, **train_params)
        yield (train_loader, 'coco')


def get_coco2014_eval_dataloaders(batch_size, text_tokenizer, img_reso=224, collate_fn=collate_fn30, pretrain=False):
    eval_list = [
        'data/coco2014/annotations/albef_coco_test.json'
    ]
    for eval_file in eval_list:
        eval_dataset = COCODataset(eval_file, 'data/coco2014', 'val', text_tokenizer, img_reso=img_reso, pretrain=pretrain)
        num_workers = 4 if img_reso > 320 else 16
        eval_params = {
                        'collate_fn': collate_fn,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'drop_last': False,
                        'num_workers': num_workers}
        eval_loader = DataLoader(eval_dataset, **eval_params)
        yield (eval_loader, 'coco')
