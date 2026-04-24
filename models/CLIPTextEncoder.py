import torch
import torch.nn as nn
import clip
from typing import List, Dict, Tuple

class CLIPTextEncoder(nn.Module):
    """
    Encodes a BATCH of text description lists using a pretrained CLIP model.
    Returns:
        - Text Features (Not rotated yet)
        - Attention Mask
        - Normalized Coordinates (Padded)
    """
    def __init__(self, clip_model_name: str = "ViT-B/16", embed_dim: int = 768):
        super().__init__()
        
        self.clip_model, _ = clip.load(clip_model_name, device="cpu", jit=False)
        self.tokenizer = clip.tokenize
        
        # Freeze all parameters of the loaded CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Projection layer to match the desired output dimension
        clip_text_dim = self.clip_model.text_projection.shape[1]
        self.text_projection = nn.Linear(clip_text_dim, embed_dim)

    def forward(self, batch_texts_with_locations: List[List[Dict]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.text_projection.weight.device
        batch_size = len(batch_texts_with_locations)

        # Flatten the batch into single lists for efficient processing
        all_texts_flat = [item['text'] for sublist in batch_texts_with_locations for item in sublist]
        all_coords_flat = [item['normalized_coord'] for sublist in batch_texts_with_locations for item in sublist]
        texts_per_image = [len(sublist) for sublist in batch_texts_with_locations]

        # Handle case where there are no texts in the entire batch
        if not all_texts_flat:
            return (
                torch.zeros(batch_size, 0, self.text_projection.out_features, device=device),
                torch.ones(batch_size, 0, dtype=torch.bool, device=device),
                torch.zeros(batch_size, 0, 2, device=device)
            )

        # Get semantic features for all texts
        tokenized_text = self.tokenizer(all_texts_flat).to(device)
        semantic_features = self.clip_model.encode_text(tokenized_text)
        projected_semantic_features = self.text_projection(semantic_features)

        # Prepare Coordinates
        coords_tensor_flat = torch.tensor(all_coords_flat, dtype=torch.float32, device=device)

        # Pad the sequences to a uniform length
        max_texts = max(texts_per_image) if texts_per_image else 0
        output_dim = projected_semantic_features.shape[-1]
        
        # Initialize outputs
        output_features = torch.zeros(batch_size, max_texts, output_dim, device=device)
        output_coords = torch.zeros(batch_size, max_texts, 2, device=device)
        
        # True means IGNORE in PyTorch MultiHeadAttention
        output_mask = torch.ones(batch_size, max_texts, dtype=torch.bool, device=device)

        feature_cursor = 0
        for i in range(batch_size):
            num_texts = texts_per_image[i]
            if num_texts > 0:
                # Fill Features
                img_feats = projected_semantic_features[feature_cursor : feature_cursor + num_texts]
                output_features[i, :num_texts] = img_feats
                
                # Fill Coords
                img_coords = coords_tensor_flat[feature_cursor : feature_cursor + num_texts]
                output_coords[i, :num_texts] = img_coords

                # Update Mask (False = Keep)
                output_mask[i, :num_texts] = False
                
                feature_cursor += num_texts

        return output_features, output_mask, output_coords