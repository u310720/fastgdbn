import logging
from functools import partial
from typing import Callable, Literal, Optional, Union

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torchinfo
from torch import nn, optim

from .constant import OUT_OF_WAFER
from .convmixer import ConvMixerBlock
from .submodules import Head, PatchEncoder, Segmenter
from .utils import DynamicWeightedBCEWithLogitsLoss, LayerNorm, create_label


class FastGDBN(L.LightningModule):
    """
    Args:
        dim: The input and output dimensions of the segmenter.
        patch_size: The height and width of the patch for the local yield distribution encoder.
        in_channels: Number of input image channels.
        sub_img_size: The desired height and width of the sub-image. If set to 'same', no padding
            is applied. Otherwise, pads the sub-image to match the specified dimensions.
        backbone: The backbone network.
        shuffle: Whether to enable patch embedding shuffling.
    """

    def __init__(
            self,
            dim: int,
            patch_size: int,
            in_channels: int = 1,
            sub_img_size: Union[Literal['same'], int] = 'same',
            backbone: Optional[Callable[[], nn.Module]] = None,
            shuffle: bool = True,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()  # Save the model architecture and its hyperparameters

        self.patch_encoder = PatchEncoder(in_channels, dim, patch_size)

        if backbone is None:
            self.segmenter = nn.Identity()
        elif shuffle is True:
            self.segmenter = Segmenter(patch_size, sub_img_size, backbone)
        else:
            self.segmenter = backbone()

        self.head = Head(dim)
        self.loss_fn = DynamicWeightedBCEWithLogitsLoss()

        self.example_input_array = torch.rand(16, 1, 25, 27)

    def forward(self, x):
        x = self.patch_encoder(x)
        x = self.segmenter(x)
        x = self.head(x)
        return x

    def on_train_start(self):
        model_info = torchinfo.summary(
            self,
            input_size=self.example_input_array.size(),
            depth=100,
            mode='train'
        )
        logging.info('\n' + str(model_info))

    def on_predict_start(self):
        model_info = torchinfo.summary(
            self,
            input_size=self.example_input_array.size(),
            depth=100,
            mode='eval'
        )
        logging.info('\n' + str(model_info))

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        logits, labels = self._common_step(batch, batch_idx)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        logits, labels = self._common_step(batch, batch_idx)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """Measures the throughput of the model."""
        logits, labels = self._common_step(batch, batch_idx)
        suspicious_levels = torch.sigmoid(logits)

    def predict_step(self, batch, batch_idx):
        logits, labels = self._common_step(batch, batch_idx)
        suspicious_levels = torch.sigmoid(logits)
        return suspicious_levels, labels

    def _common_step(self, batch: torch.Tensor, batch_idx: int):
        valid_mask = batch != OUT_OF_WAFER

        if not valid_mask.any():
            raise ValueError("No valid dies found in the batch!")

        model_outputs = self(batch)[valid_mask]
        labels = create_label(batch)[valid_mask]

        return model_outputs, labels

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=8e-4)


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
