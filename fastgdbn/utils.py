import einops
import torch
import torch.nn.functional as F
from torch import nn

from .constants import ACCEPT, BAD, GOOD, INVALID, OUT_OF_WAFER, REJECT

_VALID_INPUT_UNIQUE = torch.tensor(sorted([1, 2, 0]))

def create_label(input: torch.Tensor):
    unique_values = input.unique(sorted=True)
    expected_values = _VALID_INPUT_UNIQUE.to(input.device)

    if not torch.equal(unique_values, expected_values):
        raise RuntimeError(
            f"Input values: {unique_values.tolist()}. "
            f"Expected values: {expected_values.tolist()}."
        )

    label = torch.full_like(input, INVALID)
    label[input == BAD] = REJECT
    label[input == GOOD] = ACCEPT

    if label.dim() == 0:
        label = label.unsqueeze(0)

    return label


class DynamicWeightedBCEWithLogitsLoss(nn.Module):
    """
    Recomputes pos_weight for each batch to handle class imbalance dynamically.
    This ensures that the loss function adapts to the positive-to-negative ratio in the data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        # num_pos/num_neg: number of ones/zeros
        num_ones = target.sum()
        num_zeros = target.numel() - num_ones
        pos_weight = num_zeros / num_ones if num_ones > 0 else None
        return F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight)


class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x
