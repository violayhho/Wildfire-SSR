import sys
sys.path.append('/PATH/TO/Wildfire-SSR/models/dinov3')

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dinov3.dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from .CLIPTextEncoder import CLIPTextEncoder
from .TextEnhancedDecoder import TextEnhancedDecoder

BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class VisionLanguageSSR(nn.Module):
    def __init__(self, output_building, output_damage,
                 backbone_name="dinov3_vitl16",
                 backbone_weights='models/dinov3/dinov3/pretrained_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
                 geo_embed_type: Optional[str] = "RoPE",
                 ablated_vision: bool = False,
                 freeze_backbone=True, **kwargs):
        super(VisionLanguageSSR, self).__init__()
        assert geo_embed_type in ("RoPE", "RFF", None)
        self.geo_embed_type = geo_embed_type
        self.ablated_vision = ablated_vision

        print(f"Loading {backbone_name} from torch.hub...")
        dinov3_backbone = torch.hub.load(
            'models/dinov3', backbone_name, source="local",
            weights=backbone_weights,
        )

        self.vision_encoder = DINOv3_Adapter(
            dinov3_backbone,
            interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS[backbone_name],
        )

        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        embed_dim = self.vision_encoder.backbone.embed_dim
        patch_size = self.vision_encoder.patch_size

        self.text_encoder = CLIPTextEncoder(
            clip_model_name=kwargs.get('clip_model_name', 'ViT-B/16'),
            embed_dim=embed_dim,
        )

        self.decoder = TextEnhancedDecoder(
            patch_size=patch_size,
            encoder_dims=[embed_dim] * 4,
            num_heads=16,
            geo_embed_type=geo_embed_type,
        )

        def create_segmentation_head(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
            )

        self.building_head = create_segmentation_head(embed_dim, output_building)
        self.damage_head = create_segmentation_head(embed_dim, output_damage)

    def forward(self, rs_data, text_data):
        if self.ablated_vision:
            b, c, h, w = rs_data.shape
            embed_dim = self.vision_encoder.backbone.embed_dim
            strides = [4, 8, 16, 32]
            rs_features = []
            for stride in strides:
                seq_len = (h // stride) * (w // stride)
                empty_feature = torch.zeros((b, seq_len, embed_dim), device=rs_data.device)
                rs_features.append(empty_feature)
        else:
            rs_features_dict = self.vision_encoder(rs_data)
            rs_features = [v for k, v in sorted(rs_features_dict.items())]
            rs_features = [x.flatten(2).transpose(1, 2) for x in rs_features]

        text_features, text_mask, text_coords = self.text_encoder(
            batch_texts_with_locations=text_data
        )

        decoder_kwargs = dict(
            encoder_features=rs_features,
            text_features=text_features,
            text_mask=text_mask,
        )
        if self.geo_embed_type is not None:
            decoder_kwargs['text_coords'] = text_coords

        output = self.decoder(**decoder_kwargs)

        output_damage = self.damage_head(output)
        output_building = self.building_head(output)

        output_damage = F.interpolate(output_damage, size=rs_data.size()[-2:], mode='bilinear')
        output_building = F.interpolate(output_building, size=rs_data.size()[-2:], mode='bilinear')

        return output_building, output_damage
