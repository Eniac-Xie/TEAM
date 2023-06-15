import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

import torch
import torch.nn as nn
from tqdm import tqdm

from models.backbone.clip.model import build_model
from models.backbone.clip.simple_tokenizer import SimpleTokenizer

class CLIPVisionWrapper(nn.Module):
    def __init__(self, vision_transformer, tensor_dim=-1, use_gc=True):
        super().__init__()
        self.vision_transformer = vision_transformer
        self.vision_transformer.transformer.use_gc = use_gc

        self.tensor_dim = tensor_dim
        if self.tensor_dim > 0:
            self.tensor_fc = nn.Linear(1024, 768)
        else:
            self.tensor_fc = nn.Identity()

    def forward(self, x, return_tensor=True):
        x = self.vision_transformer.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.vision_transformer.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vision_transformer.positional_embedding.to(x.dtype)
        x = self.vision_transformer.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vision_transformer.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_tensor = x.clone()
        x_tensor = self.tensor_fc(x_tensor)
        x = self.vision_transformer.ln_post(x[:, 0, :])

        if self.vision_transformer.proj is not None:
            x = x @ self.vision_transformer.proj

        if return_tensor:
            return x, x_tensor
        else:
            return x


class CLIPTextWrapper(nn.Module):
    def __init__(self, clip_model, embed_dim=512, tensor_dim=768, use_gc=True):
        super().__init__()
        self.transformer = clip_model.transformer
        self.transformer.use_gc = use_gc
        self.vocab_size = clip_model.vocab_size
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        self.tensor_fc = nn.Linear(embed_dim, tensor_dim)

    def encode_text(self, text):
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = self.token_embedding(text)

        # x = x + self.positional_embedding.type(self.dtype)
        x = x + self.positional_embedding.type(x.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)
        x = self.ln_final(x)
        x_tensor = self.tensor_fc(x.clone())

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, x_tensor

    def forward(self, input_ids, attention_mask, output_hidden_states=False, use_linear=True, return_tensor=True):
        assert not output_hidden_states
        assert use_linear
        
        text_embedding, text_tensor = self.encode_text(input_ids)

        if return_tensor:
            return text_embedding, text_tensor
        else:
            return text_embedding

class CLIPToken:
    def __init__(self, ids, attention_mask):
        self.ids = ids.tolist()
        self.attention_mask = attention_mask.tolist()
    
class CLIPTokenizerWrapper:
    def __init__(self, ):
        self.clip_tokenizer = SimpleTokenizer()

    def encode(self, text_str):
        text_data = tokenize(self.clip_tokenizer, text_str)[0, :]
        return CLIPToken(text_data, (text_data!=0).long())

def load(model_path, vision_tensor_dim=-1, use_gc=True, large_model=True):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict()).cpu().float()
    vision_model = CLIPVisionWrapper(model.visual, tensor_dim=vision_tensor_dim, use_gc=use_gc)
    if large_model:
        text_model = CLIPTextWrapper(model, embed_dim=768, tensor_dim=1024, use_gc=use_gc)
    else:
        text_model = CLIPTextWrapper(model, use_gc=use_gc)
    return vision_model, text_model

def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    # tokenizer = SimpleTokenizer()

    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

if __name__ == '__main__':
    vision_model, text_model = load('pretrained/ViT-L-14.pt')
    tokenizer = CLIPTokenizerWrapper()
    input_tensor = torch.rand(1, 3, 224, 224)
    out, out_tensor = vision_model(input_tensor)
    print(out.shape, out_tensor.shape)

    input_text = 'a image of dogs'
    token_res = tokenizer.encode(input_text)
    text_data, text_mask = token_res.ids, token_res.attention_mask
    # import pdb;pdb.set_trace()
    # out, out_tensor = text_model(text_data[None, ...], text_mask[None, ...])
    # print(out.shape, out_tensor.shape)
