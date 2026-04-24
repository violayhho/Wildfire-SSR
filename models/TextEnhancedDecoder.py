import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math
import numpy as np
from .dinov3.dinov3.models.vision_transformer import SelfAttentionBlock as Block


class RFFGeoEmbedding(nn.Module):
    """SAM-style Random Fourier Features geolocation encoder."""
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self._pe_encoding(coords.to(torch.float))


class RoPEGeoEmbedding(nn.Module):
    """Rotary Position Embedding for geolocation coordinates (DINOv3-style)."""
    def __init__(self, embed_dim: int, base: float = 100.0, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.D_head = embed_dim // num_heads
        self.base = base
        self.register_buffer("periods", self._build_periods())

    def _build_periods(self):
        exponents = 2 * torch.arange(self.D_head // 4) / (self.D_head // 2)
        return self.base ** exponents

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_reshaped = x.view(B, N, self.num_heads, self.D_head)
        norm_coords = 2.0 * coords - 1.0
        norm_coords = norm_coords.unsqueeze(-1)

        periods_view = self.periods.view(1, -1)
        angles = 2 * math.pi * norm_coords / periods_view
        angles = angles.flatten(-2, -1)

        repeat_dims = [1] * (angles.ndim - 1) + [2]
        angles = angles.repeat(*repeat_dims)

        cos = torch.cos(angles).unsqueeze(2)
        sin = torch.sin(angles).unsqueeze(2)

        x_rotated = self.apply_rotary_pos_emb(x_reshaped, cos, sin)
        return x_rotated.reshape(B, N, C)

    def apply_rotary_pos_emb(self, x, cos, sin):
        return (x * cos) + (self.rotate_half(x) * sin)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class PatchExpand(nn.Module):
    def __init__(self, input_dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.output_dim = input_dim // 2
        self.expand = nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.norm = norm_layer(self.output_dim)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = F.pixel_shuffle(x, 2)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class DecoupledAttention(nn.Module):
    """Unified attention supporting RFF (add PE before projection), RoPE (after
    projection via `rope`), or no positional embedding (neither given)."""
    def __init__(self, query_dim, key_dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = query_dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, key_padding_mask=None,
                query_pe=None, key_pe=None,
                query_coords=None, key_coords=None, rope=None):
        B, N_q, C = query.shape
        B, N_k, _ = key.shape

        q_input = query + query_pe if query_pe is not None else query
        k_input = key + key_pe if key_pe is not None else key

        q_embed = self.q_proj(q_input)
        k_embed = self.k_proj(k_input)
        v_embed = self.v_proj(value)

        if rope is not None:
            q_embed = rope(q_embed, query_coords)
            k_embed = rope(k_embed, key_coords)

        q = q_embed.reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k_embed.reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v_embed.reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class TwoWayTextEnhancedDecoderBlock(nn.Module):
    def __init__(self, up_in_dim, skip_in_dim, text_dim, embed_dim, num_heads=8,
                 geo_embed_type: Optional[str] = "RoPE", norm_layer=nn.LayerNorm):
        super().__init__()
        assert geo_embed_type in ("RoPE", "RFF", None)
        self.geo_embed_type = geo_embed_type

        self.patch_expand = PatchExpand(input_dim=up_in_dim)
        self.concat_proj = nn.Linear((up_in_dim // 2) + skip_in_dim, embed_dim, bias=False)
        self.norm_img_in = norm_layer(embed_dim)

        self.text_adapter = nn.Linear(text_dim, embed_dim)
        self.norm_text_in = norm_layer(embed_dim)

        if geo_embed_type == "RFF":
            self.text_pe_adapter = nn.Linear(text_dim, embed_dim) if text_dim != embed_dim else nn.Identity()
        if geo_embed_type == "RoPE":
            self.rope = RoPEGeoEmbedding(embed_dim=embed_dim, base=100.0, num_heads=num_heads)

        self.text_self_attn = DecoupledAttention(embed_dim, embed_dim, num_heads=num_heads)
        self.norm_text_1 = norm_layer(embed_dim)

        self.text_to_img_attn = DecoupledAttention(embed_dim, embed_dim, num_heads=num_heads)
        self.norm_text_2 = norm_layer(embed_dim)

        self.img_to_text_attn = DecoupledAttention(embed_dim, embed_dim, num_heads=num_heads)
        self.norm_img_2 = norm_layer(embed_dim)

        self.refinement_block = Block(dim=embed_dim, num_heads=num_heads)

    def forward(self, upsampled_feature, skip_feature, text_features, text_mask,
                text_coords=None, text_pe=None, pos_encoder=None):
        x_img = self.patch_expand(upsampled_feature)
        x_img = torch.cat([x_img, skip_feature], dim=-1)
        x_img = self.concat_proj(x_img)
        x_img = self.norm_img_in(x_img)

        if torch.all(text_mask):
            x_img = self.refinement_block(x_img)
            return x_img

        B, N_img, C = x_img.shape
        H = W = int(N_img ** 0.5)
        device = x_img.device

        img_coords = None
        img_pe = None
        curr_text_pe = None

        if self.geo_embed_type is not None:
            y_coords = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
            x_coords_img = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
            gy, gx = torch.meshgrid(y_coords, x_coords_img, indexing='ij')
            img_coords = torch.stack((gx, gy), dim=-1).flatten(0, 1).unsqueeze(0).expand(B, -1, -1)

        x_text = self.text_adapter(text_features)
        x_text = self.norm_text_in(x_text)

        if self.geo_embed_type == "RFF":
            img_pe = pos_encoder(img_coords)
            curr_text_pe = self.text_pe_adapter(text_pe)

        attn_kwargs_self = {}
        attn_kwargs_t2i = {}
        attn_kwargs_i2t = {}
        if self.geo_embed_type == "RFF":
            attn_kwargs_self = dict(query_pe=curr_text_pe, key_pe=curr_text_pe)
            attn_kwargs_t2i = dict(query_pe=curr_text_pe, key_pe=img_pe)
            attn_kwargs_i2t = dict(query_pe=img_pe, key_pe=curr_text_pe)
        elif self.geo_embed_type == "RoPE":
            attn_kwargs_self = dict(query_coords=text_coords, key_coords=text_coords, rope=self.rope)
            attn_kwargs_t2i = dict(query_coords=text_coords, key_coords=img_coords, rope=self.rope)
            attn_kwargs_i2t = dict(query_coords=img_coords, key_coords=text_coords, rope=self.rope)

        attn_out = self.text_self_attn(
            query=x_text, key=x_text, value=x_text,
            key_padding_mask=text_mask, **attn_kwargs_self
        )
        x_text = x_text + attn_out
        x_text = self.norm_text_1(x_text)

        attn_out = self.text_to_img_attn(
            query=x_text, key=x_img, value=x_img,
            key_padding_mask=None, **attn_kwargs_t2i
        )
        x_text = x_text + attn_out
        x_text = self.norm_text_2(x_text)

        attn_out = self.img_to_text_attn(
            query=x_img, key=x_text, value=x_text,
            key_padding_mask=text_mask, **attn_kwargs_i2t
        )
        x_img = x_img + attn_out
        x_img = self.norm_img_2(x_img)

        x_img = x_img + self.refinement_block(x_img)
        return x_img


class TextEnhancedDecoder(nn.Module):
    def __init__(self, patch_size=16, encoder_dims=[384, 768, 1536, 3072], num_heads=12,
                 num_classes=5, geo_embed_type: Optional[str] = "RoPE"):
        super().__init__()
        assert geo_embed_type in ("RoPE", "RFF", None)
        self.encoder_dims = encoder_dims
        self.num_stages = len(encoder_dims)
        self.text_dim = encoder_dims[0]
        self.geo_embed_type = geo_embed_type

        if geo_embed_type == "RFF":
            self.pos_encoder = RFFGeoEmbedding(num_pos_feats=self.text_dim // 2, scale=1.0)

        self.decoder_blocks = nn.ModuleList()
        for i in range(self.num_stages - 1, 0, -1):
            self.decoder_blocks.append(
                TwoWayTextEnhancedDecoderBlock(
                    up_in_dim=encoder_dims[i],
                    skip_in_dim=encoder_dims[i - 1],
                    text_dim=self.text_dim,
                    embed_dim=encoder_dims[i - 1],
                    num_heads=num_heads,
                    geo_embed_type=geo_embed_type,
                )
            )

    def forward(self, encoder_features: List[torch.Tensor], text_features: torch.Tensor,
                text_mask: torch.Tensor, text_coords: Optional[torch.Tensor] = None):
        x = encoder_features[-1]

        text_pe = None
        if self.geo_embed_type == "RFF":
            text_pe = self.pos_encoder(text_coords)

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_feature = encoder_features[self.num_stages - 2 - i]

            x = decoder_block(
                upsampled_feature=x,
                skip_feature=skip_feature,
                text_features=text_features,
                text_mask=text_mask,
                text_coords=text_coords,
                text_pe=text_pe,
                pos_encoder=self.pos_encoder if self.geo_embed_type == "RFF" else None,
            )

        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x
