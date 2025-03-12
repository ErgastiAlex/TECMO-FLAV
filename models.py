# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import einops

import torch.utils.checkpoint as checkpoint

from transformers import PreTrainedModel
import random

class MelPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, n_mels, n_frames, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (n_mels // patch_size) * (n_frames // patch_size)
        self.patch_size = patch_size
        self.num_patches = int(num_patches)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class SelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            is_causal: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.is_causal = is_causal
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=self.is_causal
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            mask_attn=False,
    ):
        super().__init__()
        self.mask_attn = mask_attn
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wkv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, N, C = x.shape
        
        q = self.wq(x)
        q = einops.rearrange(q, 'B N (H D) -> B H N D', H=self.num_heads)
        
        kv = self.wkv(cond) # BMD
        kv = einops.rearrange(kv, 'B N (K H D) ->K B H N D', H=self.num_heads, K=2)
        k = kv[0]
        v = kv[1]


        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = einops.rearrange(x, 'B H N D -> B N (H D)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def temporalModulate(x, shift, scale):
    """
    Modulate the input tensor x with the given shift and scale tensors.
    :param x: the input tensor to modulate with shape (B, T, L, D).
    :param shift: the shift tensor with shape (B, T, D).
    :param scale: the scale tensor with shape (B, T, D).
    """
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class AudioEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, n_mels, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_mels, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # TODO: Activation? 

    def forward(self, a):
        a = self.mlp(a)
        return a
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 0)

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core FLAV Model                                #
#################################################################################

class FLAVBlock(nn.Module):
    """
    A FLAV block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, grad_ckpt=False, causal_attn=False, **block_kwargs):
        super().__init__()

        self.video_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.audio_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.video_audio_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.video_spatial_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.video_temporal_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, is_causal=causal_attn, **block_kwargs)
        
        self.audio_spatial_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, is_causal=causal_attn, **block_kwargs)
        # self.audio_temporal_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, is_causal=causal_attn, **block_kwargs)
        
        self.video_norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.audio_norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.video_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.audio_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.video_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.audio_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.video_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.audio_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

        self.video_scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

        self.audio_scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

        self.v_avg_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.a_avg_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
        )



        self.grad_ckpt = grad_ckpt

    def forward(self,v, a, v_c, a_c):
        if self.grad_ckpt:
            return checkpoint.checkpoint(self._forward, v, a, v_c, a_c, use_reentrant=False)
        else:
            return self._forward(v, a, v_c, a_c)

    def _forward(self, v, a, v_c, a_c):
        """
            v: Size of (B, T, Lv, D)
            a: Size of (B, T, La, D)
            v_c: Size of (B, T, D)
            a_c: Size of (B, T, D)
        """

        video_shift_msa, video_scale_msa, video_gate_msa, video_shift_tmsa, video_scale_tmsa, video_gate_tmsa = self.video_adaLN_modulation(v_c).chunk(6, dim=-1)
        # audio_shift_msa, audio_scale_msa, audio_gate_msa, audio_shift_tmsa, audio_scale_tmsa, audio_gate_tmsa = self.audio_adaLN_modulation(a_c).chunk(6, dim=-1)
        audio_shift_msa, audio_scale_msa, audio_gate_msa = self.audio_adaLN_modulation(a_c).chunk(3, dim=-1)
        B, T, L, D = v.shape

        v_att = temporalModulate(self.video_norm1(v), video_shift_msa, video_scale_msa)   
        v_att = einops.rearrange(v_att, 'B T L D -> (B T) L D')
        v_att = v + video_gate_msa.unsqueeze(2)*(self.video_spatial_attn(v_att).view(B, T, L, D))
        
        v = v_att

        v_att = temporalModulate(self.video_norm2(v_att), video_shift_tmsa, video_scale_tmsa)
        v_att = einops.rearrange(v_att, 'B T L D -> (B L) T D', T=T)
        v_att = einops.rearrange(self.video_temporal_attn(v_att), "(B L) T D -> B T L D", B=B)
        v = v + video_gate_tmsa.unsqueeze(2)*v_att

        a_att = temporalModulate(self.audio_norm1(a), audio_shift_msa, audio_scale_msa)
        a_att = einops.rearrange(a_att, 'B T L D -> B (T L) D')
        a_att = a + audio_gate_msa.unsqueeze(2)*(self.audio_spatial_attn(a_att).view(B, T, -1, D))

        a = a_att

        a_avg = self.a_avg_proj(a.mean(dim=2)) # B T D
        v_avg = self.v_avg_proj(v.mean(dim=2)) # B T D

        v_avg += a_c
        a_avg += v_c

        scale_v, shift_v, gate_v = self.video_scale(a_avg).chunk(3, dim=-1)
        scale_a, shift_a, gate_a = self.audio_scale(v_avg).chunk(3, dim=-1)


        v = v + gate_v.unsqueeze(2) * self.video_mlp(temporalModulate(self.video_norm3(v), shift_v, scale_v))
        a = a + gate_a.unsqueeze(2) * self.audio_mlp(temporalModulate(self.audio_norm3(a), shift_a, scale_a))

        return v, a
    
    def _spatial_attn(self, x, b_size, attn_func):
        x = einops.rearrange(x, "(B N) T D -> (B T) N D", B=b_size)
        x = attn_func(x)
        x = einops.rearrange(x, "(B T) N D -> (B N) T D", B=b_size)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of FLAV.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = temporalModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FLAV(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        latent_size=None,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        predict_frames = 1,
        grad_ckpt = False,
        n_mels=256,
        audio_fr = 16000,
        causal_attn = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.predict_frames = predict_frames
        self.grad_ckpt = grad_ckpt
        self.n_mels = n_mels
        self.audio_fr = audio_fr
        self.latent_size = latent_size # T H W

        self.num_classes = num_classes

        self.v_embedder = PatchEmbed(latent_size, patch_size, in_channels, hidden_size, bias=True)
        self.a_embedder = nn.Linear(n_mels, hidden_size, bias=True)

        self.video_t_embedder = TimestepEmbedder(hidden_size)
        self.audio_t_embedder = TimestepEmbedder(hidden_size)

        if self.num_classes > 0:
            self.video_y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            self.audio_y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.v_embedder.num_patches            
        self.video_spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, hidden_size), requires_grad=True)
        self.video_temporal_pos_embed = nn.Parameter(torch.zeros(1, self.predict_frames, 1, hidden_size), requires_grad=True)
        
        self.audio_spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, 10, hidden_size), requires_grad=True)
        self.audio_temporal_pos_embed = nn.Parameter(torch.zeros(1, self.predict_frames, 1, hidden_size), requires_grad=True)
        

        self.blocks = nn.ModuleList([
            FLAVBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, grad_ckpt=grad_ckpt, causal_attn=causal_attn) for _ in range(depth)
        ])

        self.video_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.audio_final_layer = FinalLayer(hidden_size, 1, n_mels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.v_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.v_embedder.proj.bias, 0)
        
        
        if self.num_classes > 0:
            nn.init.normal_(self.video_y_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.audio_y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.video_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.video_t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.audio_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.audio_t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in FLAV blocks:
        for block in self.blocks:
            nn.init.constant_(block.video_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.video_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.audio_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.audio_adaLN_modulation[-1].bias, 0)
            
            nn.init.constant_(block.video_scale[-1].weight, 0)
            nn.init.constant_(block.video_scale[-1].bias, 0)

            nn.init.constant_(block.audio_scale[-1].weight, 0)
            nn.init.constant_(block.audio_scale[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.video_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.video_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.video_final_layer.linear.weight, 0)
        nn.init.constant_(self.video_final_layer.linear.bias, 0)

        nn.init.constant_(self.audio_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.audio_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.audio_final_layer.linear.weight, 0)
        nn.init.constant_(self.audio_final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.v_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _apply_rnd_mask(self, input, mask, device="cuda"):
        input_rnd = torch.rand(input[0].shape).unsqueeze(0).to(device=device)*2 - 1
        return self._apply_mask(input, mask, input_rnd)

    def _apply_zero_mask(self, input, mask, device="cuda"):
        input_zero= torch.zeros(input[0].shape).unsqueeze(0).to(device=device)
        return self._apply_mask(input, mask, input_zero)
    
    def _get_frames_mask(self, bs):
        """
        bs: batch size
        
        returns a boolean mask to be applied to condition frames
        to mask a selected number of random frames
        """
        fmask = np.full(self.cond_frames*bs, False)
        frames = list(range(self.cond_frames))
        for b in range(bs):
            if random.randint(0, 100) < self.mask_freq:
                sub_frames = random.sample(frames, min(self.cond_frames, self.frames_to_mask))
                idxs = [f+(b*self.cond_frames) for f in sub_frames]
                fmask[idxs] = True
        return fmask
    
    def _get_batch_mask(self, bs):
        """
        bs: batch size
        
        returns a boolean mask to be applied to condition frames
        to mask a random number of condition sequences in a batch
        """
        rnd = np.random.rand(bs)
        bmask= rnd < self.batch_mask_freq/100
        bmask = np.repeat(bmask, self.cond_frames)
        return bmask
        
    def _apply_mask(self, input, mask, values):
        input[mask] = values
        return input
    
    def audio_unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        audio: (N, N_mels, frames)
        """
        c = 1
        p = self.audio_patch_size
        h = int(self.n_mels//p)
        w = int((self.audio_fr/1600)/p)


        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        audio = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return audio

    def forward(self, v, a, t, y):
        """
        Forward pass of FLAV.
        v: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        a: (B, 1, n_bins, T) # mel spectrogram of audio
        t: (B, T) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """

        ### Video
        B, T, C, H, W = v.shape
        v = einops.rearrange(v, 'B T C H W -> (B T) C H W')
        v = self.v_embedder(v)
        v = einops.rearrange(v, '(B T) L D -> B T L D', T=T)
        v = v + self.video_temporal_pos_embed + self.video_spatial_pos_embed


        ### Audio
        a = einops.rearrange(a, "B T C N F -> B T C F N").squeeze(2)
        a = self.a_embedder(a)
        a = a + self.audio_temporal_pos_embed + self.audio_spatial_pos_embed

        ### Conditioning 
        t = t.view(-1)                  # B T -> (B T)
        v_t = self.video_t_embedder(t)  # (B, T, D)
        v_t = v_t.view(B, T, -1)        # (B T) D -> B T D

        if self.num_classes > 0:
            v_y = self.video_y_embedder(y, self.training) # (B, D)
            v_y = v_y.unsqueeze(1).expand(-1, T, -1)      # (B, T, D)

        v_c = (v_t + v_y) if self.num_classes > 0 else v_t  # (B, T, D)

        a_t = self.audio_t_embedder(t) # (B, T, D)
        a_t = a_t.view(B, T, -1)

        if self.num_classes > 0:
            a_y = self.audio_y_embedder(y, self.training)
            a_y = a_y.unsqueeze(1).expand(-1, T, -1)

        a_c = (a_t + a_y) if self.num_classes > 0 else a_t    # (B, T, D)
        
        for block in self.blocks:
            v, a = block(v, a, v_c, a_c)                      # (B, T, D)
        
        v = self.video_final_layer(v, v_c)                # (B, T, patch_size ** 2 * out_channels), (B, T, L)
        a = self.audio_final_layer(a, a_c)

        v = einops.rearrange(v, 'B T L D -> (B T) L D', T = T)
        v = self.unpatchify(v)                   # (B, out_channels, H, W)
        v = einops.rearrange(v, '(B T) C H W -> B T C H W', T = T)

        a = einops.rearrange(a, 'B T F N -> B T N F', T = T).unsqueeze(2)
        return v, a

    def forward_with_cfg(self, v, a, t, y, cfg_scale):
        """
        Forward pass of FLAV, but also batches the unconditional forward pass for classifier-free guidance.
        """
        v_combined = torch.cat([v, v], dim=0)

        a_combined = torch.cat([a, a], dim=0)

        y_null = torch.tensor([self.num_classes]*v.shape[0], device=v.device)
        y = torch.cat([y, y_null], dim=0)

        t = torch.cat([t, t], dim=0)

        v_model_out, a_model_out = self.forward(v_combined, a_combined, t, y)
        v_eps = v_model_out
        a_eps = a_model_out

        v_cond_eps, v_uncond_eps = torch.split(v_eps, len(v_eps) // 2, dim=0)
        v_eps = v_uncond_eps + cfg_scale * (v_cond_eps - v_uncond_eps)

        a_cond_eps, a_uncond_eps = torch.split(a_eps, len(a_eps) // 2, dim=0)
        a_eps = a_uncond_eps + cfg_scale * (a_cond_eps - a_uncond_eps)

        return v_eps, a_eps

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/video_pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    video_pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    video_pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        video_pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), video_pos_embed], axis=0)
    return video_pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   FLAV Configs                                  #
#################################################################################

def FLAV_XL_2(**kwargs):
    return FLAV(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def FLAV_XL_4(**kwargs):
    return FLAV(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def FLAV_XL_8(**kwargs):
    return FLAV(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# def FLAV_L_2(**kwargs):
#     return FLAV(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def FLAV_L_1(**kwargs):
     return FLAV(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def FLAV_L_2(**kwargs):
    return FLAV(depth=20, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def FLAV_L_4(**kwargs):
    return FLAV(depth=20, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def FLAV_L_8(**kwargs):
    return FLAV(depth=20, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# def FLAV_B_2(**kwargs):
#     return FLAV(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def FLAV_B_1(**kwargs):
    return FLAV(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def FLAV_B_2(**kwargs):
    return FLAV(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def FLAV_B_4(**kwargs):
    return FLAV(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def FLAV_B_8(**kwargs):
    return FLAV(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def FLAV_S_2(**kwargs):
    return FLAV(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def FLAV_S_4(**kwargs):
    return FLAV(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def FLAV_S_8(**kwargs):
    return FLAV(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


FLAV_models = {
    'FLAV-XL/2':  FLAV_XL_2,  'FLAV-XL/4': FLAV_XL_4,  'FLAV-XL/8': FLAV_XL_8,
    'FLAV-L/1' :  FLAV_L_1,   'FLAV-L/2':  FLAV_L_2,   'FLAV-L/4':  FLAV_L_4,   'FLAV-L/8':  FLAV_L_8,
    'FLAV-B/1' :  FLAV_B_1,   'FLAV-B/2':  FLAV_B_2,   'FLAV-B/4':  FLAV_B_4,   'FLAV-B/8':  FLAV_B_8,
    'FLAV-S/2' :  FLAV_S_2,   'FLAV-S/4':  FLAV_S_4,   'FLAV-S/8':  FLAV_S_8,
}

