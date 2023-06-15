import os
import torch.nn as nn
from transformers import BertForMaskedLM


class BertWrapper(nn.Module):
    def __init__(self, hf_bert_dir, projector='linear', use_gradient_ckpt=False, use_cls_token=False, language='en', feat_dim=512, token_dim=-1):
        super(BertWrapper, self).__init__()
        if language == 'en':
            #  available at https://huggingface.co/bert-base-uncased/tree/main
            if not os.path.exists(hf_bert_dir):
                raise FileNotFoundError

            self.bert = BertForMaskedLM.from_pretrained(hf_bert_dir).bert
        else:
            raise ValueError()

        self.use_cls_token = use_cls_token
        
        if projector == 'linear':
            self.projector = nn.Linear(768, feat_dim, bias=False)
        elif projector == 'linear_bias':
            self.projector = nn.Linear(768, feat_dim)
        elif projector == 'iden':
            self.projector = nn.Identity()
        else:
            raise ValueError
        
        if token_dim > 0:
            self.projector_token_embeds = nn.Linear(768, token_dim)
        else:
            self.projector_token_embeds = nn.Identity()


    def forward(self, input_ids, attention_mask, return_tensor=False):
        trans_features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        output_states = self.bert(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        if self.use_cls_token:
            text_embeddings = output_tokens[:, 0, :]  # CLS token is first token
        else:
            # use mean pooling
            raise NotImplementedError
            
        if return_tensor:
            return self.projector(text_embeddings), self.projector_token_embeds(output_tokens)
        else:
            return self.projector(text_embeddings)


if __name__ == '__main__':
    model = BertWrapper(hf_bert_dir='xxx')
