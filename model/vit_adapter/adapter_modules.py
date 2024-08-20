from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from timm.models.layers import DropPath

from ..tranformer_decoder import *


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(h1, w1, h2, w2, device):
    spatial_shapes = torch.as_tensor([(h1 // 8, w1 // 8),
                                      (h1 // 16, w1 // 16),
                                      (h1 // 32, w1 // 32)],
                                     dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h2, w2)], device)
    # reference_points = get_reference_points([(h1 // 8, w1 // 8),
    #                                          (h1 // 16, w1 // 16),
    #                                          (h1 // 32, w1 // 32)], device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h2, w2)], dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h1 // 8, w1 // 8),
                                             (h1 // 16, w1 // 16),
                                             (h1 // 32, w1 // 32)], device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


def deform_inputsv2(h1, w1, h2, w2, device):
    spatial_shapes = torch.as_tensor([(h1 // 8, w1 // 8),
                                      (h1 // 16, w1 // 16),
                                      (h1 // 32, w1 // 32)],
                                     dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # reference_points = get_reference_points([(h2, w2)], device)
    reference_points = get_reference_points([(h1 // 8, w1 // 8),
                                             (h1 // 16, w1 // 16),
                                             (h1 // 32, w1 // 32)], device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h2, w2)], dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h1 // 8, w1 // 8),
                                             (h1 // 16, w1 // 16),
                                             (h1 // 32, w1 // 32)], device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, norm_layer=nn.SyncBatchNorm, use_c1_proj=True):
        super().__init__()
        self.with_cp = with_cp
        self.use_c1_proj = use_c1_proj

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        if use_c1_proj:
            self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            if self.use_c1_proj:
                c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
    
            bs, dim, _, _ = c2.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False, v_pre_norm=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        if v_pre_norm:
            self.value_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(embed_dims=dim, num_levels=n_levels, num_heads=num_heads, dropout=drop,
                                                 num_points=n_points, value_proj_ratio=deform_ratio, batch_first=True)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        self.v_pre_norm = v_pre_norm
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W, query_pos=None):
        
        def _inner_forward(query, feat):
            dtype = query.dtype
            if self.v_pre_norm:
                value = self.value_norm(feat)
            else:
                value = feat
            self.attn.float()
            attn = self.attn(query=query.to(torch.float), query_pos=query_pos, 
                             value=value.to(torch.float), key_padding_mask=None, 
                             reference_points=reference_points.to(torch.float), spatial_shapes=spatial_shapes,
                             level_start_index=level_start_index)
            query = attn.to(dtype)
            query = self.query_norm(query)
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(query, H, W))
                query = self.ffn_norm(query)
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(embed_dims=dim, num_levels=n_levels, num_heads=num_heads, dropout=0.,
                                 num_points=n_points, value_proj_ratio=deform_ratio, batch_first=True)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):
            dtype = query.dtype
            self.attn.float()
            attn = self.attn(query=self.query_norm(query).to(torch.float), identity=torch.zeros_like(query, dtype=torch.float32, requires_grad=False), 
                             value=feat.to(torch.float), key_padding_mask=None,
                             reference_points=reference_points.to(torch.float), spatial_shapes=spatial_shapes,
                             level_start_index=level_start_index,)
            return query + self.gamma * attn.to(dtype)
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, lang_dim, vl_dim=1024, num_heads=6, vl_heads=16, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, with_cp=False, num_extra_layers=-1):
        super().__init__()
        self.num_extra_layers = num_extra_layers
        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                         norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=False,
                                         cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp, v_pre_norm=True)
        self.vl_extractor = VLBiAttnLayer(dim, lang_dim, vl_dim, vl_heads, n_levels=3, mlp_ratio=cffn_ratio, dropout=drop, norm_layer=norm_layer, with_gamma=True, with_post_norm=True)
        if num_extra_layers > 0:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=False,
                            cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp, v_pre_norm=True)
                for i in range(num_extra_layers)
            ])   
            self.extra_vl_extractors = nn.Sequential(*[
                VLBiAttnLayer(dim, lang_dim, vl_dim, vl_heads, n_levels=3, mlp_ratio=cffn_ratio,  dropout=drop, norm_layer=norm_layer, with_gamma=True, with_post_norm=True)
                for i in range(num_extra_layers)
            ])
        else:
            self.extra_extractors = None
            self.extra_vl_extractors = None

    def forward(self, vit_feats, adapter_feats, lvl_pos_emb, lang_feats, lang_mask, prompts, blocks, deform_inputs1, deform_inputs2, H, W):
        b, h, w, c = vit_feats.shape
        vit_feats = self.injector(query=vit_feats.flatten(1,2), reference_points=deform_inputs1[0],
                          feat=adapter_feats, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        vit_feats = vit_feats.reshape(b, h, w, c)

        for idx, blk in enumerate(blocks):
            vit_feats = blk(vit_feats)

        adapter_feats = self.extractor(query=adapter_feats, reference_points=deform_inputs2[0],
                                             feat=vit_feats.flatten(1,2), spatial_shapes=deform_inputs2[1],
                                             level_start_index=deform_inputs2[2], H=H, W=W)

        adapter_feats, lang_feats, prompts = self.vl_extractor(adapter_feats, lang_feats, prompts, lang_mask=lang_mask, 
                                                                                 reference_points=deform_inputs2[0], spatial_shapes=deform_inputs1[1], level_start_index=deform_inputs1[2], vis_pos=lvl_pos_emb)

        if self.num_extra_layers > 0:
            for i in range(self.num_extra_layers):
                adapter_feats = self.extra_extractors[i](query=adapter_feats, reference_points=deform_inputs2[0],
                                                    feat=vit_feats.flatten(1,2), spatial_shapes=deform_inputs2[1],
                                                    level_start_index=deform_inputs2[2], H=H, W=W)
                adapter_feats, lang_feats, prompts = self.extra_vl_extractors[i](adapter_feats, lang_feats, prompts, lang_mask=lang_mask, 
                                                                                 reference_points=deform_inputs2[0], spatial_shapes=deform_inputs1[1], level_start_index=deform_inputs1[2], vis_pos=lvl_pos_emb)

        return vit_feats, adapter_feats, lang_feats, prompts