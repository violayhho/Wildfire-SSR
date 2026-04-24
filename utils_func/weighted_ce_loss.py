import torch
import torch.nn.functional as F

def distance_weighted_ce_loss(logits, targets, class_weights=None, ignore_index=255, alpha=0.5):
    """
    Computes Cross-Entropy with an added MSE distance penalty for ordinal classes.
    """
    ce_loss = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=ignore_index)
    
    # Expected Class Regression (Ordinal Distance Penalty)
    probs = F.softmax(logits, dim=1)
    
    # Create a tensor of class indices: e.g., [0., 1., 2., 3., 4.]
    classes = torch.arange(logits.size(1), device=logits.device).float()
    
    # Calculate expected class via dot product of probabilities and class indices
    # probs shape: [B, C, H, W], classes shape: [C] -> output: [B, H, W]
    expected_class = torch.einsum('b c h w, c -> b h w', probs, classes)
    
    valid_mask = (targets != ignore_index)
    
    if valid_mask.any():
        mse_loss = F.mse_loss(expected_class[valid_mask], targets[valid_mask].float())
    else:
        mse_loss = torch.tensor(0.0, device=logits.device)
        
    # Combine CE with the ordinal MSE penalty scaled by alpha
    combined_loss = ce_loss + (alpha * mse_loss)
    
    return combined_loss