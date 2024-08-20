import torch
import torch.nn as nn
from transformers import BertModel
from transformers import CLIPTextModel

from .models import *
from .segment_anything import sam_model_registry
from .criterion import criterion_dict


def _segm_refersam(pretrained, args, criterion):
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    if args.clip_path:
        text_model = CLIPTextModel.from_pretrained(args.clip_path, torch_dtype=torch_dtype)
    else:
        text_model = BertModel.from_pretrained(args.ck_bert, torch_dtype=torch_dtype)

    sam_configs = {
        "vit_h": {"interaction_indexes": [[0, 7], [8, 15], [16, 23], [24, 31]], "num_heads":16, "vl_dim":1280},
        "vit_l": {"interaction_indexes": [[0, 5], [6, 11], [12, 17], [18, 23]], "num_heads":16, "vl_dim":1024},
        "vit_b": {"interaction_indexes": [[0, 2], [3, 5], [6, 8], [9, 11]], "num_heads":12, "vl_dim":768},
    }
    sam_model = sam_model_registry[args.sam_type](pretrained)
    adapter_configs = {
        'drop_path_rate':0.0,
        'dropout':0.0,
        'conv_inplane':64,
        'n_points':4,
        'deform_num_heads':sam_configs[args.sam_type]["num_heads"],
        'deform_ratio':0.5,
        'cffn_ratio':2.0,
        'add_vit_feature':False,
        'interaction_indexes':sam_configs[args.sam_type]["interaction_indexes"],
        'with_cp': False,
        'init_values': 1e-6,
        'vl_dim': sam_configs[args.sam_type]["vl_dim"],
        'num_prompts': [16, 4],
        'num_extra_layers': 2,
        "num_prompt_layers": 2,
    }
    model = ReferSAM(sam_model, text_model, args, criterion=criterion, **adapter_configs)
    return model


def refersam(pretrained='', args=None):
    criterion = criterion_dict['mask']()
    return _segm_refersam(pretrained, args, criterion)