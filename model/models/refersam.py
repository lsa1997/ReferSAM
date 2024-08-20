import torch
from torch import nn
from torch.nn import functional as F

from ..vit_adapter import *

class ReferSAM(nn.Module):
    def __init__(self, sam_model, text_encoder, args, num_classes=1, criterion=None, **kwargs):
        super(ReferSAM, self).__init__()
        self.sam_prompt_encoder = sam_model.prompt_encoder
        self.sam_mask_decoder = sam_model.mask_decoder
        self.text_encoder = text_encoder
        self.vis_dim = sam_model.image_encoder.embed_dim
        self.lang_dim = self.text_encoder.config.hidden_size
        self.decoder_dim = self.sam_mask_decoder.transformer_dim

        self.vl_adapter = ViTAdapter(sam_model.image_encoder, self.vis_dim, lang_dim=self.lang_dim, with_deconv=True, using_clip=bool(args.clip_path),**kwargs)
        self.mask_embedding = nn.Sequential(nn.Linear(self.decoder_dim, self.decoder_dim), 
                                          nn.GELU(), 
                                          nn.Linear(self.decoder_dim, self.decoder_dim))
        self.mask_scaling = nn.Conv2d(1, 1, kernel_size=1)
        self.sparse_embedding = nn.Sequential(
                nn.Linear(self.decoder_dim, self.decoder_dim),
                nn.GELU(), 
                nn.Linear(self.decoder_dim, self.decoder_dim),
                nn.LayerNorm(self.decoder_dim, eps=1e-6),
                )

        self.using_clip = bool(args.clip_path)
        self.num_classes = num_classes
        self.criterion = criterion
        self.base_lr = args.lr
        nn.init.constant_(self.mask_scaling.weight, 1.)
        nn.init.constant_(self.mask_scaling.bias, 0.)
        self.mask_embedding.apply(self._init_weights)
        self.sparse_embedding.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def params_to_optimize(self):
        # parameters to optimize
        trainable_param_names_sam_vit = [""]
        if self.using_clip:
            trainable_param_names = ["vl_adapter", 
                                    "mask_embedding", "mask_scaling", "sparse_embedding",
                                    "mask_downscaling", "sam_mask_decoder"]
        else:
            trainable_param_names = [""]
        names_frozen = list()
        names_learnable = list()
        params_learnable = list()
        for name, m in self.named_parameters():
            if "vis_model" in name:
                if any([x in name for x in trainable_param_names_sam_vit]):
                    m.requires_grad = True
                    names_learnable.append(name)
                    params_learnable.append(m)
                else:
                    m.requires_grad = False
                    names_frozen.append(name)
            elif any([x in name for x in trainable_param_names]):
                m.requires_grad = True
                names_learnable.append(name)
                params_learnable.append(m)
            else:
                m.requires_grad = False
                names_frozen.append(name)

        print('LEARNABLE params: ', names_learnable)
        return params_learnable
    
    def encode_prompt(self, embedding_size, masks=None, text_embeds=None):
        bs = embedding_size[0]
        spatial_shape = (embedding_size[-2], embedding_size[-1])
        sparse_embeddings = torch.empty(
            (bs, 0, self.sam_prompt_encoder.embed_dim), device=self.sam_prompt_encoder._get_device()
        )

        if text_embeds is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeds], dim=1)

        if masks is not None:
            dense_embeddings = self.sam_prompt_encoder._embed_masks(masks)
        else:
            dense_embeddings = self.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, spatial_shape[0], spatial_shape[1]
            )
        dense_pe = self.sam_prompt_encoder.pe_layer(spatial_shape).unsqueeze(0)
        return sparse_embeddings, dense_embeddings, dense_pe

    def forward(self, img, text, l_mask, targets=None, return_probs=False):
        '''
            Input:
                img       [BxCxHxW]
                text    [BxN_l]
                l_mask  [BxN_l]
        '''
        batch_size = img.shape[0]
        input_shape = img.shape[-2:]

        # Text encoding
        if self.using_clip:
            with torch.no_grad():
                l_feats = self.text_encoder(text, l_mask)[0] # l_feats: [B, N_l, 768]
        else:
            l_feats = self.text_encoder(text, l_mask)[0] # l_feats: [B, N_l, 768]
        # VL pixel decoder
        adapter_feats_list, vit_feats, l_feats, all_prompts = self.vl_adapter(img, l_feats, l_mask) # vit_feats:[B, C, H/16, W/16]

        mask_feature = adapter_feats_list[0] # [B, C, H, W]
        dense_prompts = all_prompts[:, :1] # [B, 1, C]
        sparse_prompts = all_prompts[:, 1:] # [B, P, C]

        dense_prompts = self.mask_embedding(dense_prompts)
        coarse_masks = torch.einsum('bqc,bchw->bqhw', dense_prompts, mask_feature) # [B, 1, H, W]
        mask_prompt = self.mask_scaling(coarse_masks)

        sparse_prompts = self.sparse_embedding(sparse_prompts)
        sparse_embeddings, dense_embeddings, dense_pe = self.encode_prompt(vit_feats.shape, masks=mask_prompt, text_embeds=sparse_prompts)

        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=vit_feats,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings.to(dense_embeddings.dtype),
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_masks = low_res_masks.float()
        coarse_masks = coarse_masks.float()

        masks = F.interpolate(low_res_masks, size=input_shape, mode='bilinear', align_corners=True)
        pred_masks = masks.squeeze(1) # [B, H, W]
        coarse_masks = coarse_masks.squeeze(1) # [B, H, W]

        if self.training:
            if self.criterion is not None:
                losses = self.criterion(pred_masks, targets, coarse_masks)
                return losses

        if not return_probs:
            pred_masks = pred_masks.sigmoid()
            pred_masks = (pred_masks >= 0.5).long()     
        return pred_masks
