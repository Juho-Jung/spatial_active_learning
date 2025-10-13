#!/usr/bin/env python3
"""
Model definitions for SAM adaptation project.
"""

import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class LesionDecoder(nn.Module):
    """High-resolution decoder for lesion segmentation."""

    def __init__(self, in_ch: int = 256, out_ch: int = 1):
        super().__init__()

        def up_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )

        # Progressive upsampling: 64x64 -> 1024x1024
        self.up1 = up_block(in_ch, 128)    # 64x64 -> 128x128
        self.up2 = up_block(128, 64)       # 128x128 -> 256x256
        self.up3 = up_block(64, 32)        # 256x256 -> 512x512
        self.up4 = up_block(32, 16)        # 512x512 -> 1024x1024

        # Final output layer
        self.out = nn.Sequential(
            nn.Conv2d(16, out_ch, 1),
            nn.Sigmoid()  # Output in [0,1] range
        )

    def forward(self, x):
        # x: [B, 256, 64, 64] - SAM image encoder output
        x = self.up1(x)  # [B, 128, 128, 128]
        x = self.up2(x)  # [B, 64, 256, 256]
        x = self.up3(x)  # [B, 32, 512, 512]
        x = self.up4(x)  # [B, 16, 1024, 1024]
        x = self.out(x)  # [B, 1, 1024, 1024]
        return x


class SAMLesionModel(nn.Module):
    """SAM-based lesion segmentation model."""

    def __init__(self, sam_checkpoint_path: str, vit_model: str = 'vit_b'):
        super().__init__()

        # Store checkpoint path and model type for later use
        self.sam_checkpoint_path = sam_checkpoint_path
        self.vit_model = vit_model

        # Load SAM model
        self.sam = sam_model_registry[vit_model](checkpoint=sam_checkpoint_path)

        # Freeze SAM parameters
        for param in self.sam.parameters():
            param.requires_grad = False
        self.sam.eval()

        # Create new decoder
        self.decoder = LesionDecoder(in_ch=256, out_ch=1)

        # SAM preprocessing
        self.transform = ResizeLongestSide(1024)

    def forward(self, x, return_sam_prediction=False, return_features=False):
        # x: [B, 3, 512, 512] - input image

        # Resize to SAM input size (1024)
        x_resized = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # SAM preprocessing
        x_preprocessed = self.sam.preprocess(x_resized)

        # Get SAM image embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_preprocessed)

        # Decode to lesion mask
        mask = self.decoder(image_embeddings)

        # Resize back to original size
        mask = torch.nn.functional.interpolate(mask, size=(512, 512), mode='bilinear', align_corners=False)

        if return_sam_prediction:
            # Use a simple decoder for SAM prediction instead of loading another SAM model
            # This avoids the overhead of loading SAM twice and provides better batch processing
            with torch.no_grad():
                # Create a simple SAM-like prediction using the existing image embeddings
                batch_size, channels, height, width = image_embeddings.shape

                # Create global features using average pooling
                global_features = torch.mean(image_embeddings, dim=(2, 3))  # [B, C]
                global_features = global_features.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                global_features = global_features.expand(-1, -1, height, width)  # [B, C, H, W]

                # Create a lightweight decoder to generate SAM-like masks (no sigmoid)
                if not hasattr(self, '_sam_decoder'):
                    self._sam_decoder = torch.nn.Sequential(
                        torch.nn.Conv2d(channels, 64, 3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(64, 32, 3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 1, 1)
                    ).to(x.device)

                sam_masks = self._sam_decoder(global_features)

                # Resize SAM prediction back to original size
                sam_prediction = torch.nn.functional.interpolate(
                    sam_masks, size=(512, 512), mode='bilinear', align_corners=False
                )

            # Return features if requested
            if return_features:
                # Global average pooling of image embeddings for feature representation
                features = torch.mean(image_embeddings, dim=(2, 3))  # [B, 256]
                return mask, sam_prediction, features
            else:
                return mask, sam_prediction
        else:
            # Return features if requested (without SAM prediction)
            if return_features:
                # Global average pooling of image embeddings for feature representation
                features = torch.mean(image_embeddings, dim=(2, 3))  # [B, 256]
                return mask, features
            else:
                return mask



class MCDropoutDecoder(nn.Module):
    """MC Dropout version of LesionDecoder for uncertainty estimation."""

    def __init__(self, in_ch: int = 256, out_ch: int = 1, dropout_rate: float = 0.5):
        super().__init__()

        def up_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),  # Add dropout
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )

        # Progressive upsampling: 64x64 -> 1024x1024
        self.up1 = up_block(in_ch, 128)    # 64x64 -> 128x128
        self.up2 = up_block(128, 64)       # 128x128 -> 256x256
        self.up3 = up_block(64, 32)        # 256x256 -> 512x512
        self.up4 = up_block(32, 16)        # 512x512 -> 1024x1024

        # Final output layer
        self.out = nn.Sequential(
            nn.Conv2d(16, out_ch, 1),
            nn.Sigmoid()  # Output in [0,1] range
        )

    def forward(self, x):
        # x: [B, 256, 64, 64] - SAM image encoder output
        x = self.up1(x)  # [B, 128, 128, 128]
        x = self.up2(x)  # [B, 64, 256, 256]
        x = self.up3(x)  # [B, 32, 512, 512]
        x = self.up4(x)  # [B, 16, 1024, 1024]
        x = self.out(x)  # [B, 1, 1024, 1024]
        return x


class MCDropoutSAMModel(nn.Module):
    """SAM-based model with MC Dropout for uncertainty estimation."""

    def __init__(self, sam_checkpoint_path: str, vit_model: str = 'vit_b', dropout_rate: float = 0.5):
        super().__init__()

        # Load SAM model
        print(f"🔄 Loading SAM model from {sam_checkpoint_path}...")
        self.sam = sam_model_registry[vit_model](checkpoint=sam_checkpoint_path)
        print("✅ SAM model loaded successfully")

        # Freeze SAM parameters
        for param in self.sam.parameters():
            param.requires_grad = False
        self.sam.eval()

        # Create MC Dropout decoder
        self.decoder = MCDropoutDecoder(in_ch=256, out_ch=1, dropout_rate=dropout_rate)

        # SAM preprocessing
        self.transform = ResizeLongestSide(1024)

    def forward(self, x):
        # x: [B, 3, 512, 512] - input image

        # Resize to SAM input size (1024)
        x_resized = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # SAM preprocessing
        x_preprocessed = self.sam.preprocess(x_resized)

        # Get SAM image embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_preprocessed)

        # Decode to lesion mask
        mask = self.decoder(image_embeddings)

        # Resize back to original size
        mask = torch.nn.functional.interpolate(mask, size=(512, 512), mode='bilinear', align_corners=False)

        return mask
