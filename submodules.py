import logging
import math
from typing import Callable, Literal, Union

import einops
import torch
import torch.nn.functional as F
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size,
            padding='same',
        )
        self.act_layer = act_layer()
        self.norm_layer = norm_layer(out_channels)

    def forward(self, x):
        _, _, h, w = self.conv.weight.size()

        with torch.no_grad():
            self.conv.weight[:, :, h // 2, w // 2] = 0

        x = self.conv(x)
        x = self.act_layer(x)
        x = self.norm_layer(x)
        return x


class Segmenter(nn.Module):
    """
    Args:
        patch_size: The height and width of the patch for the local yield distribution encoder.
        sub_img_size: The desired height and width of the sub-image. If set to 'same', no padding
            is applied. Otherwise, pads the sub-image to match the specified dimensions.
        backbone: The segmentation algorithm.
    """

    def __init__(
            self,
            patch_size: int,
            sub_img_size: Union[Literal['same'], int] = 'same',
            backbone: Callable[[], nn.Module] = nn.Identity,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.sub_img_size = sub_img_size

        self.partition = nn.PixelUnshuffle(patch_size)
        self.backbone = backbone()
        self.reassemble = nn.PixelShuffle(patch_size)

        self.remove_batch_norm()

    def forward(self, x: torch.Tensor):
        p = self.patch_size
        _, _, h, w = x.size()
        h_, w_ = self.make_divisible(h, p), self.make_divisible(w, p)

        x = F.pad(x, (0, w_ - w, 0, h_ - h))
        x = einops.rearrange(x, 'b c h w -> b c 1 h w')
        x = self.partition(x)
        x = einops.rearrange(x, 'b c pp h w -> (b pp) c h w', pp=p**2)

        if self.sub_img_size == 'same':
            x = self.backbone(x)
        else:
            _, _, sub_h, sub_w = x.size()
            target_h = target_w = self.sub_img_size

            if sub_h > target_h or sub_w > target_w:
                raise ValueError(
                    f"Sub-image size ({sub_h}, {sub_w}) is larger than target size ({target_h}, {target_w})."
                )

            x = F.pad(x, (0, target_w - sub_w, 0, target_h - sub_h))
            x = self.backbone(x)
            x = x[:, :, :sub_h, :sub_w]

        x = einops.rearrange(x, '(b pp) c h w -> b c pp h w', pp=p**2)
        x = self.reassemble(x)
        x = einops.rearrange(x, 'b c 1 h w -> b c h w')
        x = x[:, :, :h, :w]
        return x

    @staticmethod
    def make_divisible(x, divisor):
        return math.ceil(x / divisor) * divisor

    def remove_batch_norm(self, warning: bool = True):
        def inner(model: nn.Module):
            for name, module in model.named_children():
                if isinstance(module, nn.BatchNorm2d):
                    if warning is True:
                        logging.warning(
                            f"Removing BatchNorm layer {module}.{name}"
                        )
                    setattr(model, name, nn.Identity())
                else:
                    inner(module)
            return model
        return inner(self)


class Head(nn.Sequential):
    def __init__(self, dim: int, dropout: float = 0.):
        super().__init__(
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, 1, kernel_size=1)
        )
