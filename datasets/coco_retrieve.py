import json
import io
import os
import random
import tempfile
import re

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, \
        RandomHorizontalFlip, RandomResizedCrop
        
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from datasets.coco import pre_caption


class COCORetrieveDataset(Dataset):
    def __init__(self, ann_file, image_root, max_words=80, transform=None, img_reso=224, text_processor=pre_caption):
        self.ann = json.load(open(ann_file,'r'))

        if transform is None:
            self.transform = Compose(
                        [
                            Resize((img_reso, img_reso), interpolation=Image.BICUBIC),
                            ToTensor(),
                            Normalize(
                                (0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711))
                        ])
        else:
            self.transform = transform  

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
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        oss_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(oss_path).convert('RGB')
        image = self.transform(image)
        return image, index

def get_coco2014_retrieve_dataloaders(batch_size, img_reso=224):
    eval_list = [
        'data/coco2014/annotations/albef_coco_test.json'
    ]
    for eval_file in eval_list:
        eval_dataset = COCORetrieveDataset(eval_file, 'data/coco2014', img_reso=img_reso)
        eval_params = {
                        'batch_size': batch_size,
                        'shuffle': False,
                        'drop_last': False,
                        'num_workers': 16}
        eval_loader = DataLoader(eval_dataset, **eval_params)
        yield (eval_loader, 'coco_retrieve')

def get_flickr_retrieve_dataloaders(batch_size, img_reso=224):
    eval_list = [
        'data/flickr30k/flickr30k_test_indent.json'
    ]
    for eval_file in eval_list:
        eval_dataset = COCORetrieveDataset(eval_file, 'data/flickr30k', img_reso=img_reso)
        eval_params = {
                        'batch_size': batch_size,
                        'shuffle': False,
                        'drop_last': False,
                        'num_workers': 16}
        eval_loader = DataLoader(eval_dataset, **eval_params)
        yield (eval_loader, 'flickr_retrieve')

if __name__ == '__main__':
    pass
