import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

def PosEncoding(feats, pos):
    if pos is not None:
        return feats + pos
    else:
        return feats


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None, act='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU() if act =='gelu' else nn.ReLU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self._reset_parameters()

    def extra_repr(self):
        return 'num_heads={}'.format(self.num_heads)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        trunc_normal_(self.q_proj.weight)
        trunc_normal_(self.k_proj.weight)
        trunc_normal_(self.v_proj.weight)
        trunc_normal_(self.out_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz) # [B, num_heads, src_len, head_dim]
        value_states = self._shape(self.v_proj(v), -1, bsz) # [B, num_heads, src_len, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # [B*num_heads, tgt_len, head_dim]
        key_states = key_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        value_states = value_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # [B*num_heads, tgt_len, src_len]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, src_len]
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            # attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        # return attn_output, attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_weights


class MultiScaleAttention(nn.Module):
    """
    Multi-scale multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, num_levels=1, dropout=0.1):
        super(MultiScaleAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.level_weights = nn.Linear(self.q_dim, self.num_heads*self.num_levels)

        self._reset_parameters()

    def extra_repr(self):
        return 'num_heads={}'.format(self.num_heads)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        trunc_normal_(self.q_proj.weight, std=.02)
        trunc_normal_(self.k_proj.weight, std=.02)
        trunc_normal_(self.v_proj.weight, std=.02)
        trunc_normal_(self.out_proj.weight, std=.02)
        trunc_normal_(self.level_weights.weight, std=.02)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.out_proj.bias.data.fill_(0)
        self.level_weights.bias.data.fill_(0)
    
    def ms_attention(self, q, k, v, level_weights, k_spatial_shapes, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()
        key_states = self._shape(k, -1, bsz) # [B, num_heads, src_len, head_dim]
        value_states = self._shape(v, -1, bsz) # [B, num_heads, src_len, head_dim]

        q_states = self._shape(q, tgt_len, bsz) # [B, num_heads, tgt_len, head_dim]

        src_len = key_states.size(2)
        attn_weights = torch.matmul(q_states, key_states.transpose(2, 3)) # [B, num_heads, tgt_len, src_len]
        
        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, src_len]
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights_list = attn_weights.split([H_ * W_ for H_, W_ in k_spatial_shapes], dim=-1)
        level_weights = level_weights.reshape(bsz, tgt_len, self.num_heads, 1, self.num_levels).transpose(1, 2) # [B, num_heads, tgt_len, 1, num_levels]
        level_weights = level_weights.softmax(-1)
        attn_weights = torch.cat([nn.functional.softmax(attn_weights_list[i], dim=-1) * level_weights[..., i] for i in range(self.num_levels)], dim=-1)

        attn_probs = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_probs, value_states) # [B, num_heads, tgt_len, head_dim]

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        return attn_output, attn_weights

    def forward(self, q, k, v, k_spatial_shapes, attention_mask=None, return_attention=False):
        level_weights = self.level_weights(q)
        q = self.q_proj(q) * self.scale
        k = self.k_proj(k)
        v = self.v_proj(v)

        attn_output, attn_weights = self.ms_attention(q, k, v, level_weights, k_spatial_shapes, attention_mask, return_attention)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).view(x.size(0), -1, self.num_pos_feats*2) # [B,(HxW),C]
        return pos


class VLBiAttnLayer(nn.Module):
    def __init__(self, v_dim, l_dim, inner_dim=1024, heads=16, n_levels=1, mlp_ratio=2,
                 dropout=0., norm_layer=nn.LayerNorm, with_gamma=False, init_value=1.0, with_post_norm=True):
        super().__init__()
        self.with_post_norm = with_post_norm
        self.l2v_cross_attn = MultiScaleAttention(l_dim, v_dim, inner_dim, num_heads=heads, num_levels=n_levels, dropout=dropout)
        self.p2l_cross_attn = MultiHeadAttention(l_dim, l_dim, inner_dim, num_heads=heads, dropout=dropout)
        self.v2l_cross_attn = MultiHeadAttention(v_dim, l_dim, inner_dim, num_heads=heads, dropout=dropout)

        self.prompt_mlp = FeedForward(l_dim, int(l_dim*mlp_ratio), dropout, act='gelu')
        self.vis_mlp = FeedForward(v_dim, int(v_dim*mlp_ratio), dropout, act='gelu')

        self.with_gamma = with_gamma
        if with_gamma:
            self.gamma_v2l = nn.Parameter(init_value*torch.ones((1, 1, v_dim)), requires_grad=True)
            self.gamma_l2v = nn.Parameter(init_value*torch.ones((1, 1, l_dim)), requires_grad=True)

        self.v_norm1 = norm_layer(v_dim)
        if self.with_post_norm:
            self.v_norm2 = norm_layer(v_dim)

        self.v_drop1 = nn.Dropout(dropout)
        self.v_drop2 = nn.Dropout(dropout)

        self.l_norm1 = norm_layer(l_dim)
        self.l_norm2 = norm_layer(l_dim)
        self.l_norm3 = norm_layer(l_dim)

        self.l_drop1 = nn.Dropout(dropout)
        self.l_drop2 = nn.Dropout(dropout)
        self.l_drop3 = nn.Dropout(dropout)
        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, vis, lang, prompts, lang_mask, spatial_shapes, vis_pos=None, vis_padding_mask=None, **kwargs):
        '''
            - vis: :math:`(N, L, E)` where L is the sequence length, N is the batch size, E is
            the embedding dimension.
            - lang: :math:`(N, S, E)`, where S is the sequence length, N is the batch size, E is
            the embedding dimension.
            - prompts: :math:`(N, P, E)` where P is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - lang_mask :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the zero positions will be ignored while the position
            with the non-zero positions will be unchanged.
            - vis_pos: :math:`(N, L, E)` where L is the sequence length, N is the batch size, E is
            the embedding dimension.
            - vis_padding_mask :math:`(N, L)` where N is the batch size, L is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged.
        '''
        batch_size = lang.shape[0]
        text_length = lang.shape[1]
        if prompts is not None:
            prompted_lang = torch.cat([lang, prompts], dim=1)
            prompted_lang_mask = torch.cat([lang_mask, lang_mask.new_ones((batch_size, prompts.shape[1]))], dim=1)
        else:
            prompted_lang = lang
            prompted_lang_mask = lang_mask
        if vis_padding_mask is None:
            vis_padding_mask = vis.new_zeros((vis.shape[0], vis.shape[1]))

        # Lang self attn
        _prompted_lang = self.p2l_cross_attn(q=prompted_lang, k=prompted_lang, v=prompted_lang, attention_mask=prompted_lang_mask)[0]
        prompted_lang = prompted_lang + self.l_drop2(_prompted_lang)
        prompted_lang = self.l_norm2(prompted_lang)

        # L2V cross attn
        _prompted_lang = self.l2v_cross_attn(q=prompted_lang, k=PosEncoding(vis, vis_pos), v=vis, k_spatial_shapes=spatial_shapes, attention_mask=(1-vis_padding_mask.byte()))[0]
        if self.with_gamma:
            _prompted_lang = _prompted_lang * self.gamma_l2v
        prompted_lang = prompted_lang + self.l_drop1(_prompted_lang)
        prompted_lang = self.l_norm1(prompted_lang)

        # Lang FFN
        _prompted_lang = self.prompt_mlp(prompted_lang)
        prompted_lang = prompted_lang + self.l_drop3(_prompted_lang)
        prompted_lang = self.l_norm3(prompted_lang)

        # V2L cross attn
        _vis = self.v2l_cross_attn(q=PosEncoding(vis, vis_pos), k=prompted_lang, v=prompted_lang, attention_mask=prompted_lang_mask)[0]
        if self.with_gamma:
            _vis = _vis * self.gamma_v2l
        vis = vis + self.v_drop1(_vis)
        vis = self.v_norm1(vis)

        # Visual FFN
        _vis = self.vis_mlp(vis)
        vis = vis + self.v_drop2(_vis)
        if self.with_post_norm:
            vis = self.v_norm2(vis)

        if prompts is not None:
            lang = prompted_lang[:, :text_length]
            prompts = prompted_lang[:, text_length:]
        else:
            lang = prompted_lang

        return vis, lang, prompts
    

class PromptAttnLayer(nn.Module):
    def __init__(self, p_dim, v_dim, l_dim, heads=16, n_levels=3, mlp_ratio=4,
                 dropout=0., norm_layer=nn.LayerNorm,):
        super().__init__()
        self.p2v_cross_attn = MultiScaleAttention(p_dim, v_dim, p_dim, num_heads=heads, num_levels=n_levels, dropout=dropout)
        self.p2l_cross_attn = MultiHeadAttention(p_dim, l_dim, p_dim, num_heads=heads, dropout=dropout)
        self.p_self_attn = MultiHeadAttention(p_dim, p_dim, p_dim, num_heads=heads, dropout=dropout)
        self.prompt_mlp = FeedForward(p_dim, int(p_dim*mlp_ratio), dropout, act='gelu')

        self.norm1 = norm_layer(p_dim)
        self.norm2 = norm_layer(p_dim)
        self.norm3 = norm_layer(p_dim)
        self.norm4 = norm_layer(p_dim)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.drop4 = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, prompts, vis, lang, lang_mask, spatial_shapes, prompt_pos=None, vis_pos=None, vis_padding_mask=None):
        if vis_padding_mask is None:
            vis_padding_mask = vis.new_zeros((vis.shape[0], vis.shape[1]))

        _prompts = self.p_self_attn(q=PosEncoding(prompts, prompt_pos), k=PosEncoding(prompts, prompt_pos), v=prompts)[0]
        prompts = prompts + self.drop1(_prompts)
        prompts = self.norm1(prompts)

        _prompts = self.p2v_cross_attn(q=PosEncoding(prompts, prompt_pos), k=PosEncoding(vis, vis_pos), v=vis, k_spatial_shapes=spatial_shapes, attention_mask=(1-vis_padding_mask.byte()))[0]
        prompts = prompts + self.drop2(_prompts)
        prompts = self.norm2(prompts)

        _prompts = self.p2l_cross_attn(q=PosEncoding(prompts, prompt_pos), k=lang, v=lang, attention_mask=lang_mask)[0]
        prompts = prompts + self.drop3(_prompts)
        prompts = self.norm3(prompts)

        # prompt FFN
        _prompts = self.prompt_mlp(prompts)
        prompts = prompts + self.drop4(_prompts)
        prompts = self.norm4(prompts)

        return prompts