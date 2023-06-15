import time
import torch
import torch.nn.functional as F
import numpy as np
from datasets.coco import pre_caption


def padding_tokens_worker(tokens_list, MAX_LEN):
    max_text_len = 0
    for tokens in tokens_list:
        if len(tokens.ids) > max_text_len:
            max_text_len = len(tokens.ids)
    if max_text_len <= MAX_LEN:
        max_text_len = MAX_LEN
    else:
        raise ValueError

    text_code_tensor = torch.zeros((len(tokens_list), max_text_len)).long()
    text_mask_tensor = torch.zeros((len(tokens_list), max_text_len))

    for idx, tokens in enumerate(tokens_list):
        text, text_mask = tokens.ids, tokens.attention_mask
        text_code_tensor[idx, 0:len(text)] = torch.tensor(text)
        text_mask_tensor[idx, 0:len(text_mask)] = torch.tensor(text_mask)

    return text_code_tensor, text_mask_tensor


def padding_tokens_30(tokens_list):
    return padding_tokens_worker(tokens_list, 30)

def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    assert not np.any(np.isnan(scores_i2t))
    assert not np.any(np.isnan(scores_t2i))

    #Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def evaluation_retrieve(model, data_loader, tokenizer, device, two_stage, padding_tokens):
    # test
    model.to(device)
    model.eval()
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds, text_tensors, text_masks = [], [], []
    time_s = time.time()
    with torch.no_grad():
        for i in range(0, num_text, text_bs):
            print('extracting {}/{} text embedding'.format(i, num_text))
            sub_texts = texts[i: min(num_text, i+text_bs)]
            tokens_list = []
            for text in sub_texts:
                tokens = tokenizer.encode(pre_caption(text, max_words=30))
                tokens_list.append(tokens)
            
            text_ids, text_mask = padding_tokens(tokens_list)
            text_embed, text_tensor = model.text_model(text_ids.to(device), attention_mask=text_mask.to(device), return_tensor=True)
            text_embed = F.normalize(text_embed, p=2.0, dim=1)
            text_embeds.append(text_embed)
            text_tensors.append(text_tensor)
            text_masks.append(text_mask)
        text_embeds = torch.cat(text_embeds, dim=0)
        text_tensors = torch.cat(text_tensors, dim=0)
        text_masks = torch.cat(text_masks, dim=0)
    time_e = time.time()
    print('one-stage text feature takes: {} sec'.format(time_e - time_s))
    
    image_embeds, image_tensors = [], []
    time_s = time.time()
    with torch.no_grad():
        for idx, (image, img_id) in enumerate(data_loader): 
            print('extracting {}/{} image embedding'.format(idx, len(data_loader)))
            image = image.to(device) 
            image_embed, image_tensor = model.image_model(image, return_tensor=True)
            image_embed = F.normalize(image_embed, p=2.0, dim=1)     
            image_embeds.append(image_embed)
            image_tensors.append(image_tensor.cpu())
        image_embeds = torch.cat(image_embeds, dim=0)
        
        if image_tensors[0] is not None:
            image_tensors = torch.cat(image_tensors, dim=0)
    time_e = time.time()
    print('one-stage image feature takes: {} sec'.format(time_e - time_s))
    score_matrix_i2t = image_embeds @ text_embeds.t()
    score_matrix_t2i = text_embeds @ image_embeds.t()

    # 第一阶段
    print('evaluate first stage')
    eval_result = itm_eval(
        score_matrix_i2t.cpu().numpy(),
        score_matrix_t2i.cpu().numpy(),
        data_loader.dataset.txt2img,
        data_loader.dataset.img2txt
    )
    print(eval_result)

    if not two_stage:
        return

    # 第二阶段
    score_matrix_i2t_refine = torch.zeros_like(score_matrix_i2t) * 0.0 - 1000.0
    score_matrix_t2i_refine = torch.zeros_like(score_matrix_t2i) * 0.0 - 1000.0

    topk=16
    with torch.no_grad():
        time_s = time.time()
        for imgid in range(score_matrix_i2t.shape[0]):
            if imgid % 500 == 0:
                print('processing {}-th image'.format(imgid))
            _, topk_textid = torch.topk(score_matrix_i2t[imgid], k=topk)
            this_image_tensors = image_tensors[imgid].repeat(topk, 1, 1)

            match_preds = model.get_team_score(
                text_tensors[topk_textid],
                text_masks[topk_textid].to(device),
                this_image_tensors.to(device)
            )
            match_preds = torch.sum(
                torch.stack([tmp[0] for tmp in match_preds]),
                dim=0
            )
            match_preds = [match_preds, ]
            score_matrix_i2t_refine[imgid, topk_textid] = match_preds[0]
        time_e = time.time()
        print('two-stage i2t takes: {} sec'.format(time_e - time_s))

        time_s = time.time()
        for textid in range(score_matrix_t2i.shape[0]):
            if textid % 500 == 0:
                print('processing {}-th text'.format(textid))
            _, topk_imageid = torch.topk(score_matrix_t2i[textid], k=topk)
            this_text_tensors = text_tensors[textid].repeat(topk, 1, 1)
            this_text_masks = text_masks[textid].repeat(topk, 1)
            topk_image_tensors = image_tensors[topk_imageid]

            match_preds = model.get_team_score(
                this_text_tensors,
                this_text_masks.to(device),
                topk_image_tensors.to(device)
            )
            match_preds = torch.sum(
                torch.stack([tmp[0] for tmp in match_preds]),
                dim=0
            )
            match_preds = [match_preds, ]
            score_matrix_t2i_refine[textid, topk_imageid] = match_preds[0]
        time_e = time.time()
        print('two-stage t2i takes: {} sec'.format(time_e - time_s))

    score_matrix_t2i_refine = score_matrix_t2i_refine.cpu().numpy()
    score_matrix_i2t_refine = score_matrix_i2t_refine.cpu().numpy()

    print('evaluate second stage')
    eval_result = itm_eval(
        score_matrix_i2t_refine,
        score_matrix_t2i_refine,
        data_loader.dataset.txt2img,
        data_loader.dataset.img2txt
    )
    print(eval_result)
