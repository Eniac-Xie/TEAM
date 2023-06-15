import io
import os
import torch
import torch.nn as nn

from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def check_same_size(self, t1, t2):
        if len(t1.shape) != len(t2.shape):
            return False
        
        t1_shape = tuple(t1.shape)
        t2_shape = tuple(t2.shape)
        for s1, s2 in zip(t1_shape, t2_shape):
            if s1 != s2:
                return False
        
        return True
    
    def load_weight_from_file(self, init_model, verbose=True):
        if os.path.exists(init_model):
            logger.info('found local: {}'.format(init_model))
            model_content = open(init_model, 'rb').read()
        else:
            raise FileNotFoundError

        src_params = torch.load(io.BytesIO(model_content), map_location='cpu')
        new_src_params = {}
        for k, v in src_params.items():
            if k[:7] == 'module.':
                new_src_params[k[7:]] = v
            else:
                new_src_params[k] = v
        
        dest_params = self.state_dict()
        loaded_keys = {}
        for k, v in dest_params.items():
            if k in new_src_params:
                if not self.check_same_size(dest_params[k], new_src_params[k]):
                    logger.warning('unmatch shape of {}, skip init it'.format(k))
                    continue
                dest_params[k].copy_(new_src_params[k])
                loaded_keys[k] = None
                if verbose:
                    logger.info('loading {} from init_model'.format(k))
            else:
                logger.warning('missing {}'.format(k))

        for k in new_src_params:
            if k not in loaded_keys:
                logger.warning('ignore {} in init_model'.format(k))


    def forward_test(self, *args):
        raise NotImplementedError
    
    def forward_backward(self, *args):
        raise NotImplementedError
