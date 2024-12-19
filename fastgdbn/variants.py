import math
from functools import partial

import segmentation_models_pytorch as smp
from torch import nn

from .convmixer import ConvMixerBlock
from .fastgdbn import FastGDBN
from .utils import LayerNorm


def convmixer_fastgdbn(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs) -> FastGDBN:
    decoder_depth = encoder_depth
    return FastGDBN(
        dim, patch_size,
        sub_img_size='same',
        backbone=partial(
            nn.Sequential,
            *[
                ConvMixerBlock(
                    dim,
                    kernel_size=3,
                    norm_layer=partial(LayerNorm, dim)
                )
                for _ in range(encoder_depth + decoder_depth)
            ]
        ),
        *args, **kwargs
    )


def deeplabv3_fastgdbn_bms(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs) -> FastGDBN:
    # The maximum allowable wafer size is determined by patch_size * sub_img_size
    # In BM-S, the height and width are (25, 27)
    # To meet the requirement of divisibility by 8, patch_size is set to 7 and sub_img_size to 8
    return FastGDBN(
        dim, patch_size,
        sub_img_size=8,
        backbone=partial(
            smp.DeepLabV3,
            encoder_name="efficientnet-b0",
            encoder_depth=encoder_depth,
            decoder_channels=dim * 2**encoder_depth,
            in_channels=dim,
            classes=dim,
            activation=nn.GELU
        ),
        *args, **kwargs
    )


def deeplabv3_fastgdbn_bml_bm1b(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs) -> FastGDBN:
    # The maximum allowable wafer size is determined by patch_size * sub_img_size
    # In the WM-811K dataset, the maximum height and width are (300, 205)
    # To meet the requirement of divisibility by 8, patch_size is set to 7 and sub_img_size to 48
    return FastGDBN(
        dim, patch_size,
        sub_img_size=48,
        backbone=partial(
            smp.DeepLabV3,
            encoder_name="efficientnet-b0",
            encoder_depth=encoder_depth,
            decoder_channels=dim * 2**encoder_depth,
            in_channels=dim,
            classes=dim,
            activation=nn.GELU
        ),
        *args, **kwargs
    )


def fpn_fastgdbn_bms(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs):
    # The maximum allowable wafer size is determined by patch_size * sub_img_size
    # In BM-S, the height and width are (25, 27)
    # To meet the requirement of divisibility by 8, patch_size is set to 7 and sub_img_size to 8
    return FastGDBN(
        dim, patch_size,
        sub_img_size=8,
        backbone=partial(
            smp.FPN,
            encoder_name="efficientnet-b0",
            encoder_depth=encoder_depth,
            in_channels=dim,
            decoder_segmentation_channels=2*dim,
            decoder_pyramid_channels=4*dim,
            upsampling=1,
            classes=dim
        ),
        *args, **kwargs
    )


def fpn_fastgdbn_bml_bm1b(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs):
    # The maximum allowable wafer size is determined by patch_size * sub_img_size
    # In the WM-811K dataset, the maximum height and width are (300, 205)
    # To meet the requirement of divisibility by 8, patch_size is set to 7 and sub_img_size to 48
    return FastGDBN(
        dim, patch_size,
        sub_img_size=48,
        backbone=partial(
            smp.FPN,
            encoder_name="efficientnet-b0",
            encoder_depth=encoder_depth,
            in_channels=dim,
            decoder_segmentation_channels=2*dim,
            decoder_pyramid_channels=4*dim,
            upsampling=1,
            classes=dim
        ),
        *args, **kwargs
    )


def unet_fastgdbn_bms(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs):
    # The maximum allowable wafer size is determined by patch_size * sub_img_size
    # In BM-S, the height and width are (25, 27)
    # To meet the requirement of divisibility by 8, patch_size is set to 7 and sub_img_size to 8
    return FastGDBN(
        dim, patch_size,
        sub_img_size=8,
        backbone=partial(
            smp.Unet,
            encoder_name="efficientnet-b0",
            encoder_depth=encoder_depth,
            decoder_channels=list(
                reversed([dim * 2**i for i in range(encoder_depth)])),
            in_channels=dim,
            classes=dim,
            activation=nn.GELU
        ),
        *args, **kwargs
    )


def unet_fastgdbn_bml_bm1b(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs):
    # The maximum allowable wafer size is determined by patch_size * sub_img_size
    # In the WM-811K dataset, the maximum height and width are (300, 205)
    # To meet the requirement of divisibility by 8, patch_size is set to 7 and sub_img_size to 48
    return FastGDBN(
        dim, patch_size,
        sub_img_size=48,
        backbone=partial(
            smp.Unet,
            encoder_name="efficientnet-b0",
            encoder_depth=encoder_depth,
            decoder_channels=list(
                reversed([dim * 2**i for i in range(encoder_depth)])),
            in_channels=dim,
            classes=dim,
            activation=nn.GELU
        ),
        *args, **kwargs
    )


def patch_encoder_leak_fastgdbn(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs):
    """Substitute the masked convolution with a normal convolution"""
    model = convmixer_fastgdbn(dim, patch_size, encoder_depth, *args, **kwargs)
    model.patch_encoder = nn.Sequential(
        nn.Conv2d(1, dim, patch_size, padding='same'),
        nn.GELU(),
        nn.BatchNorm2d(dim)
    )
    return model


def segmenter_leak_fastgdbn(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs):
    """Illegal sampling period"""
    model = convmixer_fastgdbn(dim, patch_size, encoder_depth, *args, **kwargs)
    legal_period_lower_bound = math.ceil(patch_size / 2)
    model.segmenter.partition = nn.PixelUnshuffle(legal_period_lower_bound - 1)
    model.segmenter.reassemble = nn.PixelShuffle(legal_period_lower_bound - 1)
    return model


def head_leak_fastgdbn(dim=32, patch_size=7, encoder_depth=3, *args, **kwargs):
    """Fusing features without the isolation of PixelUnshuffle/PixelShuffle"""
    model = convmixer_fastgdbn(dim, patch_size, encoder_depth, *args, **kwargs)
    model.head = nn.Sequential(
        nn.Conv2d(dim, dim, 3, padding='same'),
        nn.GELU(),
        model.head
    )
    return model
