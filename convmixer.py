from typing import Callable

from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Sequential):
    """See the paper: `Patches Are All You Need?`
    by Asher Trockman and Zico Kolter (2022) for more details."""

    def __init__(
            self,
            dim: int,
            kernel_size: int,
            dilation: int = 1,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.Identity
    ):
        super().__init__(
            Residual(nn.Sequential(
                nn.Conv2d(
                    dim, dim,
                    kernel_size=kernel_size,
                    padding='same',
                    dilation=dilation,
                    groups=dim,
                ),
                act_layer(),
                norm_layer()
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            act_layer(),
            norm_layer()
        )
