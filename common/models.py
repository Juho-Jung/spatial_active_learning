"""
Common model definitions for segmentation tasks.
"""

from typing import Union

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """Improved UNet for segmentation with deeper architecture."""

    def __init__(self) -> None:
        super().__init__()

        # Encoder - Deeper architecture
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(128, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(256, 512, 3, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 512, 3, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(inplace=True))

        # Decoder - Symmetric with encoder
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, stride=2),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2),
                                  nn.Conv2d(128, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1),
                                  nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with skip connections
        d4 = self.dec4(e4)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        return d1


def create_model(model_type: str = 'basic', device: Union[str, torch.device] = 'cuda') -> nn.Module:
    """
    Create a model based on the specified type.

    Args:
        model_type: Type of model to create ('basic', 'swinunetr', 'smp', 'smp_efficientnet')
        device: Device to place the model on

    Returns:
        The created model
    """
    if model_type == 'basic':
        model = SimpleUNet()
    elif model_type == 'swinunetr':
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(img_size=(512, 512),
                          in_channels=3,
                          out_channels=1,
                          feature_size=96,
                          use_checkpoint=True,
                          spatial_dims=2,
                          use_v2=True)
        # Add Sigmoid to ensure output is in [0,1] range
        model = nn.Sequential(model, nn.Sigmoid())
    elif model_type == 'smp':
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet50',
                         encoder_weights='imagenet',
                         decoder_attention_type='scse',
                         in_channels=3,
                         classes=1,
                         activation='sigmoid')
    elif model_type == 'smp_efficientnet':
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='efficientnet-b4',
                         encoder_weights='imagenet',
                         decoder_attention_type='scse',
                         in_channels=3,
                         classes=1,
                         activation='sigmoid')
    else:
        raise ValueError(f"Unknown model: {model_type}. Choose from: basic, swinunetr, smp, smp_efficientnet")

    return model.to(device)
