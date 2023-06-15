import tempfile
import torch
import numpy
import random


class MMData(object):
    def __init__(self, text, text_mask, img, misc, mlm_text=None, mlm_target=None, mlm_text_mask=None, aug_img=None):
        self.text = text
        self.text_mask = text_mask
        self.img = img
        self.misc = misc

        self.mlm_text = mlm_text
        self.mlm_target = mlm_target
        self.mlm_text_mask = mlm_text_mask
        self.aug_img = aug_img

    def to(self, device):
        self.text = self.text.to(device)
        self.text_mask = self.text_mask.to(device)
        self.img = self.img.to(device)

        if self.mlm_text is not None:
            self.mlm_text = self.mlm_text.to(device)
        if self.mlm_target is not None:
            self.mlm_target = self.mlm_target.to(device)
        if self.mlm_text_mask is not None:
            self.mlm_text_mask = self.mlm_text_mask.to(device)
        if self.aug_img is not None:
            self.aug_img = self.aug_img.to(device)
        return self


class MMBatch(object):
    def __init__(self, text, text_mask, img, misc_list, mlm_text=None, mlm_target=None, mlm_text_mask=None, aug_img=None):
        self.text = text
        self.text_mask = text_mask
        self.img = img
        self.misc_list = misc_list

        self.mlm_text = mlm_text
        self.mlm_target = mlm_target
        self.mlm_text_mask = mlm_text_mask
        self.aug_img = aug_img

    def to(self, device):
        self.text = self.text.to(device)
        self.text_mask = self.text_mask.to(device)
        self.img = self.img.to(device)

        if self.mlm_text is not None:
            self.mlm_text = self.mlm_text.to(device)
        if self.mlm_target is not None:
            self.mlm_target = self.mlm_target.to(device)
        if self.mlm_text_mask is not None:
            self.mlm_text_mask = self.mlm_text_mask.to(device)
        if self.aug_img is not None:
            self.aug_img = self.aug_img.to(device)
        return self


def collate_worker(input_list, max_text_len):
    for item in input_list:
        assert len(item.text) <= max_text_len, 'len(item.text): {}'.format(len(item.text))

    text_code_tensor = input_list[0].text.new_full((len(input_list), max_text_len), 0)
    text_mask_tensor = input_list[0].text_mask.new_full((len(input_list), max_text_len), 0)
    img_tensor = input_list[0].img.new_full((len(input_list), ) + input_list[0].img.shape, 0)
    misc_list = []
    if input_list[0].mlm_text is not None:
        mlm_text_code_tensor = input_list[0].mlm_text.new_full((len(input_list), max_text_len), 0)
    else:
        mlm_text_code_tensor = None
    if input_list[0].mlm_target is not None:
        mlm_target_tensor = input_list[0].mlm_target.new_full((len(input_list), max_text_len), -100)
    else:
        mlm_target_tensor = None
    if input_list[0].mlm_text_mask is not None:
        mlm_text_mask_tensor = input_list[0].mlm_text_mask.new_full((len(input_list), max_text_len), 0)
    else:
        mlm_text_mask_tensor = None
    if input_list[0].aug_img is not None:
        aug_img_tensor = input_list[0].aug_img.new_full((len(input_list), ) + input_list[0].aug_img.shape, 0)
    else:
        aug_img_tensor = None

    for idx, item in enumerate(input_list):
        text, text_mask, img, misc, mlm_text, mlm_target, mlm_text_mask, aug_img = item.text, item.text_mask, item.img, item.misc, item.mlm_text, item.mlm_target, item.mlm_text_mask, item.aug_img
        text_code_tensor[idx, 0:len(text)] = text
        text_mask_tensor[idx, 0:len(text_mask)] = text_mask
        img_tensor[idx] = img
        misc_list.append(misc)

        if mlm_text_code_tensor is not None:
            mlm_text_code_tensor[idx, 0:len(mlm_text)] = mlm_text
        if mlm_target_tensor is not None:
            mlm_target_tensor[idx, 0:len(mlm_target)] = mlm_target
        if mlm_text_mask_tensor is not None:
            mlm_text_mask_tensor[idx, 0:len(mlm_text_mask)] = mlm_text_mask
        if aug_img_tensor is not None:
            aug_img_tensor[idx] = aug_img

    return MMBatch(text_code_tensor, text_mask_tensor, img_tensor, misc_list, mlm_text_code_tensor, mlm_target_tensor, mlm_text_mask_tensor, aug_img_tensor)


def collate_fn30(input_list):
    return collate_worker(input_list, 30)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
