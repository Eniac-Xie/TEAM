import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist

from models.base_model import BaseModel
from models.backbone.vit import Mlp
from models.utils import GatherLayer
from utils.logging import AverageMeter
from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()


class TEAMBlock(nn.Module):
    def __init__(self, feat_dim=768, mlp_ratio=4, use_dropout=True, use_mlp=True, use_self_attn=True):
        super(TEAMBlock, self).__init__()
        self.use_dropout = use_dropout
        self.use_mlp = use_mlp
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        if self.use_mlp:
            self.norm3 = nn.LayerNorm(feat_dim)

        if self.use_self_attn:
            self.self_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=feat_dim // 64)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=feat_dim // 64)
        if self.use_mlp:
            self.ffn = Mlp(in_features=feat_dim, hidden_features=feat_dim*mlp_ratio, drop=0.1 if self.use_dropout else 0.0)

        if self.use_dropout:
            if self.use_self_attn:
                self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)
            if self.use_mlp:
                self.dropout3 = nn.Dropout(0.1)
        else:
            if self.use_self_attn:
                self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()
            if self.use_mlp:
                self.dropout3 = nn.Identity()
    
    def forward(self, text_tensors, text_masks, image_tensors, retrieved_tensors):
        if self.use_self_attn:
            retrieved_tensors_res = self.norm1(retrieved_tensors)
            retrieved_tensors_res = self.self_attn(
                (text_tensors + retrieved_tensors_res).permute(1, 0, 2),
                (text_tensors + retrieved_tensors_res).permute(1, 0, 2),
                retrieved_tensors_res.permute(1, 0, 2),
                key_padding_mask=(text_masks==0),
            )[0].permute(1, 0, 2)
            retrieved_tensors = retrieved_tensors + self.dropout1(retrieved_tensors_res)

        retrieved_tensors_res = self.norm2(retrieved_tensors)
        retrieved_tensors_res = self.cross_attn(
            (text_tensors + retrieved_tensors_res).permute(1, 0, 2),
            image_tensors.permute(1, 0, 2),
            image_tensors.permute(1, 0, 2)
        )[0].permute(1, 0, 2)
        retrieved_tensors = retrieved_tensors + self.dropout2(retrieved_tensors_res)
        
        if self.use_mlp:
            retrieved_tensors_res = self.norm3(retrieved_tensors)
            retrieved_tensors = retrieved_tensors + self.dropout3(self.ffn(retrieved_tensors_res))

        return retrieved_tensors

class TEAM(BaseModel):
    def __init__(self,
                text_model,
                image_model,
                pretrained,
                neg_num=1,
                team_block_num=6,
                mask_noise=False,
                retrieve_mlp_ratio=4,
                retrieve_use_dropout=True,
                retrieve_use_mlp=True,
                retrieve_use_self_attn=True,
                use_sep_group=False,
                layerwise_loss=True,
                token_dim=768):
        super(TEAM, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        self.team_block_num = team_block_num
        self.team_block_list = nn.ModuleList(
            [TEAMBlock(feat_dim=token_dim, mlp_ratio=retrieve_mlp_ratio, use_dropout=retrieve_use_dropout, use_mlp=retrieve_use_mlp, use_self_attn=retrieve_use_self_attn) for _ in range(self.team_block_num)]
        )

        self.image_tensor_fc = nn.Linear(token_dim, 512)
        self.text_tensor_fc = nn.Linear(token_dim, 512)

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.66)  # ln(1/0.07)=2.66
        self.logit_scale_team = nn.Parameter(torch.ones([]) * 2.66)  # ln(1/0.07)=2.66
        self.loss2_metric = AverageMeter('Loss2')
        self.loss3_metric = AverageMeter('Loss3')

        self.neg_num = neg_num
        self.mask_noise = mask_noise
        self.use_sep_group = use_sep_group
        self.layerwise_loss = layerwise_loss

        if pretrained is not None:
            logger.info('loading from {}'.format(pretrained))
            self.load_weight_from_file(pretrained)

    def get_feature(self, text_data, text_mask, img_tensor):
        text_feature, text_tensors = self.text_model(text_data, text_mask, return_tensor=True)
        text_feature = F.normalize(text_feature, p=2.0, dim=1)
        image_feature, image_tensors = self.image_model(img_tensor, return_tensor=True)
        image_feature = F.normalize(image_feature, p=2.0, dim=1)

        return text_feature, text_tensors, image_feature, image_tensors
    
    def forward(self, *args, **kwargs):
        if kwargs['phase'] == 'eval':
            batch_data, dist_info = args
            return self.forward_test(batch_data, dist_info)
        elif kwargs['phase'] == 'train':
            batch_data, dist_info, print_log, log_info = args
            node_group = kwargs['node_group']
            return self.forward_train(batch_data, dist_info, print_log, log_info, node_group)
        else:
            raise ValueError
 
    def forward_test(self, batch_data, dist_info):
        with torch.no_grad():
            text_data = batch_data.text
            text_mask = batch_data.text_mask
            img_tensor = batch_data.img

            device = torch.device('cuda:{}'.format(dist_info['local_rank']))
            text_data = text_data.to(device, non_blocking=True)
            text_mask = text_mask.to(device, non_blocking=True)
            img_tensor = img_tensor.to(device, non_blocking=True)

            text_feature, _, image_feature, _ = self.get_feature(text_data, text_mask, img_tensor)

            match_score_mat = torch.mm(text_feature, image_feature.t())
            score = match_score_mat.data.cpu().numpy().flatten()
            assert match_score_mat.shape[0] == match_score_mat.shape[1]
            gt = torch.eye(match_score_mat.shape[0]).long().numpy().flatten()
            return match_score_mat
 
    def contrastive_loss(self, logits, dim):
        neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
        return -neg_ce.mean()

    def clip_loss(self, text_image_similarity, image_text_similarity, img_idx=None, all_img_idx=None):
        if img_idx is not None and all_img_idx is not None:
            with torch.no_grad():
                neg_indicator = (img_idx[:, None] != all_img_idx[None, :])
                neg_indicator.fill_diagonal_(True)
                neg_indicator = neg_indicator.type(text_image_similarity.dtype)
            caption_loss = self.contrastive_loss(text_image_similarity * neg_indicator, dim=1)
            image_loss = self.contrastive_loss(image_text_similarity * neg_indicator, dim=1)
        else:
            caption_loss = self.contrastive_loss(text_image_similarity, dim=1)
            image_loss = self.contrastive_loss(image_text_similarity, dim=1)
        return (caption_loss + image_loss) / 2.0

    def select_neg(
                self,
                text_feature, image_feature,
                text_tensors, text_mask, image_tensors,
                all_text_feature, all_image_feature,
                all_text_tensors, all_text_mask, all_image_tensors,
                logit_scale,
                img_idx=None,
                all_img_idx=None):
        bs = text_feature.shape[0]
        with torch.no_grad():
            t2i_sim = F.softmax(text_feature @ all_image_feature.t() * logit_scale, dim=1)
            i2t_sim = F.softmax(image_feature @ all_text_feature.t() * logit_scale, dim=1)
            t2i_sim.fill_diagonal_(0)
            i2t_sim.fill_diagonal_(0)
            if img_idx is not None and all_img_idx is not None:
                pos_indicator = (img_idx[:, None] == all_img_idx[None, :])
                t2i_sim[pos_indicator] = 0
                i2t_sim[pos_indicator] = 0

        neg_image_tensors = []
        for bid in range(bs):
            if torch.any(torch.isinf(t2i_sim[bid])):
                logger.info('INF occur')
                t2i_sim[bid][torch.isinf(t2i_sim[bid])] = 0
            if torch.any(torch.isnan(t2i_sim[bid])):
                logger.info('NaN occur')
                t2i_sim[bid][torch.isnan(t2i_sim[bid])] = 0
            if torch.any(t2i_sim[bid] < 0):
                logger.info('Negative occur')
                t2i_sim[bid][t2i_sim[bid] < 0] = 0

            neg_idx = torch.multinomial(t2i_sim[bid], self.neg_num)
            neg_image_tensors.append(all_image_tensors[neg_idx])
        neg_image_tensors = torch.stack(neg_image_tensors, dim=0)  # bs * self.neg_num * L * D

        bs = image_feature.shape[0]
        neg_text_tensors = []
        neg_text_mask = []
        for bid in range(bs):
            neg_idx = torch.multinomial(i2t_sim[bid], self.neg_num)
            neg_text_tensors.append(all_text_tensors[neg_idx])
            neg_text_mask.append(all_text_mask[neg_idx])
        neg_text_tensors = torch.stack(neg_text_tensors, dim=0)
        neg_text_mask = torch.stack(neg_text_mask, dim=0)

        return neg_text_tensors, neg_text_mask, neg_image_tensors
    
    def re_arrange_tensor(self, gathered_list, world_size, global_rank):
        assert len(gathered_list) == world_size
        return [gathered_list[global_rank], ] + [gathered_list[idx] for idx in range(world_size) if idx != global_rank]

    def get_team_score(self, text_tensors, text_mask, image_tensors):
        retrieved_tensors = torch.zeros_like(text_tensors)
        pair_score_list = []
        text_tensors_proj = self.text_tensor_fc(text_tensors)
        text_mask_float = text_mask.type(text_tensors_proj.dtype)
        for each_team_block in self.team_block_list:
            retrieved_tensors = each_team_block(text_tensors, text_mask, image_tensors, retrieved_tensors)
            retrieved_tensors_proj = self.image_tensor_fc(retrieved_tensors)

            pair_score = torch.sum(F.normalize(retrieved_tensors_proj, p=2.0, dim=2) * F.normalize(text_tensors_proj, p=2.0, dim=2), dim=2)
            pair_score_reduced = torch.sum(pair_score * text_mask_float, dim=1) / torch.clamp(torch.sum(text_mask_float, dim=1), min=1.0)
            pair_score_list.append(
                (pair_score_reduced, None)
            )
        return pair_score_list

    def get_match_loss(self, pos_pair_score_list, neg_pair_score1_list, neg_pair_score2_list, logit_scale_team):
        loss3_list = []
        for pos_pair_score, neg_pair_score1, neg_pair_score2 in zip(pos_pair_score_list, neg_pair_score1_list, neg_pair_score2_list):
            pos_neg1_neg2_score = torch.cat(
                [
                    pos_pair_score[0][:, None],
                    neg_pair_score1[0],
                    neg_pair_score2[0],
                ],
                dim=1
            ) * logit_scale_team
            loss3 = -torch.mean(F.log_softmax(pos_neg1_neg2_score, dim=1)[:, 0])

            loss3_list.append(loss3)
        return loss3_list

    def forward_train(self, batch_data, dist_info, print_log, log_info, node_group):         
        text_data = batch_data.text
        text_mask = batch_data.text_mask
        img_tensor = batch_data.img

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logit_scale_team = self.logit_scale_team.exp().clamp(max=100.0)

        device = torch.device('cuda:{}'.format(dist_info['local_rank']))
        text_data = text_data.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        img_tensor = img_tensor.to(device, non_blocking=True)

        text_feature, text_tensors, image_feature, image_tensors = self.get_feature(text_data, text_mask, img_tensor)

        # neg pair
        # gather features and tensors
        if self.use_sep_group:
            dist.barrier()

        if self.mask_noise:
            batch_img_idx = batch_data.misc_list
            batch_img_idx = torch.tensor(batch_img_idx, dtype=torch.long, device=text_feature.device)
            gather_layer0 = GatherLayer.apply
            world_img_idx = gather_layer0(batch_img_idx)
            world_img_idx = torch.cat(self.re_arrange_tensor(world_img_idx, dist.get_world_size(), dist.get_rank()), dim=0)

        gather_layer1_5 = GatherLayer.apply
        tv_feature = torch.cat((text_feature, image_feature), dim=1)
        world_feature = gather_layer1_5(tv_feature)
        world_text_feature, world_image_feature = torch.split(
            torch.cat(self.re_arrange_tensor(world_feature, dist.get_world_size(), dist.get_rank()), dim=0),
            [text_feature.shape[1], image_feature.shape[1]],
            dim=1
        )

        ###########################################################
        # global embeddings alignment learning
        ###########################################################
        global_rank = dist_info['global_rank']
        text_img_sco = torch.matmul(text_feature, world_image_feature.t()) * logit_scale
        img_text_sco = torch.matmul(image_feature, world_text_feature.t()) * logit_scale

        loss2 = self.clip_loss(
            text_img_sco,
            img_text_sco,
            img_idx=batch_img_idx if self.mask_noise else None,
            all_img_idx=world_img_idx if self.mask_noise else None)
        self.loss2_metric.update(loss2.item())

        if self.use_sep_group:
            dist.barrier()

        ###########################################################
        # token embeddings alignment learning
        ###########################################################
        if self.use_sep_group:
            if self.mask_noise:
                gather_layer0_copy = GatherLayer.apply
                node_img_idx = gather_layer0_copy(batch_img_idx, node_group)
                node_img_idx = torch.cat(self.re_arrange_tensor(node_img_idx, dist.get_world_size(group=node_group), dist.get_rank(group=node_group)), dim=0)

            gather_layer1_5_copy = GatherLayer.apply
            node_feature = gather_layer1_5_copy(tv_feature, node_group)
            node_text_feature, node_image_feature = torch.split(
                torch.cat(self.re_arrange_tensor(node_feature, dist.get_world_size(group=node_group), dist.get_rank(group=node_group)), dim=0),
                [text_feature.shape[1], image_feature.shape[1]],
                dim=1
            )
        else:
            if self.mask_noise:
                node_img_idx = world_img_idx
            node_text_feature = world_text_feature
            node_image_feature = world_image_feature

        gather_layer2 = GatherLayer.apply
        text_tensors_list = gather_layer2(text_tensors, node_group if self.use_sep_group else None)
        node_text_tensors = torch.cat(self.re_arrange_tensor(text_tensors_list, dist.get_world_size(group=node_group if self.use_sep_group else None), dist.get_rank(group=node_group if self.use_sep_group else None)), dim=0)

        gather_layer3 = GatherLayer.apply
        text_mask_list = gather_layer3(text_mask, node_group if self.use_sep_group else None)
        node_text_mask = torch.cat(self.re_arrange_tensor(text_mask_list, dist.get_world_size(group=node_group if self.use_sep_group else None), dist.get_rank(group=node_group if self.use_sep_group else None)), dim=0)

        gather_layer4 = GatherLayer.apply
        image_tensors_list = gather_layer4(image_tensors, node_group if self.use_sep_group else None)
        node_image_tensors = torch.cat(self.re_arrange_tensor(image_tensors_list, dist.get_world_size(group=node_group if self.use_sep_group else None), dist.get_rank(group=node_group if self.use_sep_group else None)), dim=0)

        if self.use_sep_group:
            dist.barrier()

        # pos pair: text_tensors, image_tensors
        pos_pair_score_list = self.get_team_score(text_tensors, text_mask, image_tensors)

        # neg pair1: text_tensors, neg_image_tensors
        # neg pair2: neg_text_tensors, image_tensors
        neg_text_tensors, neg_text_mask, neg_image_tensors = self.select_neg(
                text_feature, image_feature,
                text_tensors, text_mask, image_tensors,
                node_text_feature, node_image_feature,
                node_text_tensors, node_text_mask, node_image_tensors,
                logit_scale,
                img_idx=batch_img_idx if self.mask_noise else None,
                all_img_idx=node_img_idx if self.mask_noise else None)

        neg_pair_score1_list = self.get_team_score(
            text_tensors[:, None, :, :].expand(-1, self.neg_num, -1, -1).flatten(start_dim=0, end_dim=1),
            text_mask[:, None, :].expand(-1, self.neg_num, -1).flatten(start_dim=0, end_dim=1),
            neg_image_tensors.flatten(start_dim=0, end_dim=1)
        )
        neg_pair_score1_list = [(tmp1.reshape(-1, self.neg_num), None if tmp2 is None else tmp2.reshape(-1, self.neg_num)) for tmp1, tmp2 in neg_pair_score1_list]
        neg_pair_score2_list = self.get_team_score(
            neg_text_tensors.flatten(start_dim=0, end_dim=1),
            neg_text_mask.flatten(start_dim=0, end_dim=1),
            image_tensors[:, None, :, :].expand(-1, self.neg_num, -1, -1).flatten(start_dim=0, end_dim=1)
        )
        neg_pair_score2_list = [(tmp1.reshape(-1, self.neg_num), None if tmp2 is None else tmp2.reshape(-1, self.neg_num)) for tmp1, tmp2 in neg_pair_score2_list]

        loss3_list = self.get_match_loss(pos_pair_score_list, neg_pair_score1_list, neg_pair_score2_list, logit_scale_team)
        if self.layerwise_loss:
            loss3 = torch.sum(torch.stack(loss3_list))
        else:
            loss3 = loss3_list[-1]

        self.loss3_metric.update(loss3.item())

        if print_log:
            info_str = 'rank={}, epoch={}, batch={}/{}, batch_size={}, loss2={:.4f}, loss3={:.4f}, logit_scale={:.4f}, logit_scale_team={:.4f}'.format(
                    global_rank, log_info['epoch'], log_info['batch_idx'], log_info['all_batch_cnt'], tv_feature.shape[0],
                    self.loss2_metric.avg, self.loss3_metric.avg, logit_scale.item(), logit_scale_team.item())
            logger.info(info_str)
        
        loss = loss2 + loss3
            
        return loss
        
