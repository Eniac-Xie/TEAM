# Token Embeddings Alignment for Cross-Modal Retrieval

PyTorch implementation and pretrained models of TEAM.
A new dataset which contains over 100M Chinese image-text pairs will also be released.

![Model](figs/framework.jpg)

## Pretrained Models

We provide three pre-trained models:

[pretrained_4m.pth](): TEAM with ViT-B/16 (initialized by [DeiT-base](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)) as image encoder, pre-trained on 4 millions of image-text pairs.

[pretrained_14m_clip_large.pth](): TEAM with ViT-L/14 (initialized by [CLIP-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)) as image encoder, pre-trained on 14 millions of image-text pairs.

Besides, we also release TEAM trained on our collected Chinese image-text dataset, please refere to [TEAM图文检索模型-中文-large](https://modelscope.cn/models/damo/multi-modal_team-vit-large-patch14_multi-modal-similarity/summary) for more details.

## Evaluation

To evaluate the [pretrained_14m_clip_large.pth]() on COCO Retrieval task, you can run:
```shell
python -m eval configs/pretrain_5m/team_clipl14.py
```
Note that the results of the second stage is the final results.

## Training
To train TEAM with ViT-L/14 as image encoder on **4 millions** of image-text pairs:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/pretrain_5m/team_clipl14.py
```

## Experimental Results
### COCO Retrieval
<table border="1" width="100%">
	<tr align="center">
        <th></th><th colspan="6">Zero-shot</th><th colspan="6">Finetune</th>
    </tr>
    <tr align="center">
        <th></th><th colspan="3">Text Retrieval</th><th colspan="3">Image Retrieval</th><th colspan="3">Text Retrieval</th><th colspan="3">Image Retrieval</th>
    </tr>
    <tr align="center">
        <td></td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
    <tr align="center">
        <td>pretrained_4m</td><td>74.9</td><td>91.8</td><td>95.3</td><td>54.7</td><td>79.5</td><td>86.6</td><td>77.3</td><td>93.6</td><td>96.5</td><td>59.7</td><td>83.2</td><td>89.4</td>
    </tr>
    <tr align="center">
        <td>pretrained_14m_clip_large</td><td>82.8</td><td>95.6</td><td>97.6</td><td>63.9</td><td>85.1</td><td>90.4</td><td>84.0</td><td>96.1</td><td>98.0</td><td>66.9</td><td>87.0</td><td>92.1</td>
    </tr>
</table>


## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{TEAM2022MM,
  title = {Token Embeddings Alignment for Cross-Modal Retrieval},
  author = {Xie, Chen-Wei and Wu, Jianmin and Zheng, Yun and Pan, Pan and Hua, Xian-Sheng},
  booktitle = {ACMMM},
  year = {2022}
}
```

Some code is borrowed from [ALBEF](https://github.com/salesforce/ALBEF) and [CLIP](https://github.com/openai/CLIP). Thanks a lot to them.
