#!/usr/bin/env python3
"""
Segmentation and detection models for active learning (U-Net, SAM-based, LUNIT, etc.).
"""

import torch
import torch.nn as nn

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    _SAM_AVAILABLE = True
except ImportError:
    sam_model_registry = None
    ResizeLongestSide = None
    _SAM_AVAILABLE = False


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
        print(f"Loading SAM model from {sam_checkpoint_path}...")
        self.sam = sam_model_registry[vit_model](checkpoint=sam_checkpoint_path)
        print("SAM model loaded.")

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


class SegmentationModel(nn.Module):
    """General segmentation model"""

    def __init__(self, model_type: str = 'smp_efficientnet', device: str = 'cuda'):
        super().__init__()

        self.model_type = model_type
        self.device = device

        if model_type == 'swinunetr':
            from monai.networks.nets import SwinUNETR
            self.model = SwinUNETR(img_size=(512, 512), in_channels=3, out_channels=1, feature_size=96,
                                   use_checkpoint=True, spatial_dims=2, use_v2=True).to(device)
            # Add Sigmoid to ensure output is in [0,1] range
            self.model = nn.Sequential(self.model, nn.Sigmoid()).to(device)
        elif model_type == 'smp':
            import segmentation_models_pytorch as smp
            self.model = smp.Unet(
                encoder_name='resnet50',
                encoder_weights='imagenet',
                decoder_attention_type='scse',
                in_channels=3,
                classes=1,
                activation='sigmoid'
            ).to(device)
        elif model_type == 'smp_efficientnet':
            import segmentation_models_pytorch as smp
            self.model = smp.Unet(
                encoder_name='efficientnet-b4',
                encoder_weights='imagenet',
                decoder_attention_type='scse',
                in_channels=3,
                classes=1,
                activation='sigmoid'
            ).to(device)
        else:
            raise ValueError(f"Unknown model: {model_type}. Choose from: swinunetr, smp, smp_efficientnet")

    def forward(self, x, return_sam_prediction=False, return_features=False):
        """
        Forward pass with same interface as SAMLesionModel.

        Args:
            x: Input tensor [B, 3, 512, 512]
            return_sam_prediction: For compatibility (ignored)
            return_features: Whether to return features

        Returns:
            mask: Segmentation mask [B, 1, 512, 512]
            features: Optional features [B, feature_dim] if return_features=True
        """
        # x: [B, 3, 512, 512] - input image

        # Get segmentation mask
        mask = self.model(x)  # [B, 1, 512, 512]

        if return_features:
            # For compatibility, extract features from the model
            # This is a simple approach - you might want to modify based on your needs
            if hasattr(self.model, 'encoder'):
                # For SMP models, extract encoder features
                encoder_output = self.model.encoder(x)
                # SMP encoder returns a list of features from different stages
                # Use the last (highest-level) feature map
                if isinstance(encoder_output, (list, tuple)):
                    features = encoder_output[-1]  # [B, C, H, W]
                else:
                    features = encoder_output  # [B, C, H, W]
                # Global average pooling
                features = torch.mean(features, dim=(2, 3))  # [B, feature_dim]
            else:
                # For other models, use a simple feature extraction
                with torch.no_grad():
                    # Use the last layer before sigmoid as features
                    if isinstance(self.model, nn.Sequential):
                        # For SwinUNETR with sigmoid wrapper
                        features = self.model[0](x)
                        features = torch.mean(features, dim=(2, 3))  # [B, feature_dim]
                    else:
                        # For other models, create dummy features
                        batch_size = x.shape[0]
                        features = torch.zeros(batch_size, 256, device=x.device)

            return mask, features
        else:
            return mask


class LossPredictionModule(nn.Module):
    """
    Loss Prediction Module for Learning Loss active learning.

    Takes multi-level features from the target model and predicts the loss
    value without requiring ground truth labels.
    """

    def __init__(self, feature_dims, hidden_dim=128):
        """
        Args:
            feature_dims: List of feature dimensions from different layers
            hidden_dim: Hidden dimension for FC layers
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim

        # Process each feature level: GAP -> FC -> ReLU
        self.feature_processors = nn.ModuleList()
        for dim in feature_dims:
            processor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
                nn.Flatten(),
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.feature_processors.append(processor)

        # Concatenate all processed features
        total_dim = hidden_dim * len(feature_dims)

        # Final FC layer to predict loss (scalar)
        self.fc = nn.Linear(total_dim, 1)

    def forward(self, features_list):
        """
        Args:
            features_list: List of feature maps from different layers
                Each feature map: [B, C, H, W]

        Returns:
            predicted_loss: [B, 1] - Predicted loss values
        """
        processed_features = []
        for i, features in enumerate(features_list):
            # Process each feature level
            processed = self.feature_processors[i](features)  # [B, hidden_dim]
            processed_features.append(processed)

        # Concatenate all processed features
        concat_features = torch.cat(processed_features, dim=1)  # [B, hidden_dim * num_levels]

        # Predict loss
        predicted_loss = self.fc(concat_features)  # [B, 1]

        return predicted_loss


class SegmentationModelWithLossPrediction(nn.Module):
    """
    Segmentation Model with Loss Prediction Module for LUNIT active learning.

    This wrapper adds a loss prediction module to the base segmentation model,
    enabling joint training of both the target model and loss prediction module.
    """

    def __init__(self, base_model, feature_dims=None, hidden_dim=128):
        """
        Args:
            base_model: Base segmentation model (SegmentationModel)
            feature_dims: List of feature dimensions from different layers
                         If None, will try to infer from model
            hidden_dim: Hidden dimension for loss prediction module
        """
        super().__init__()

        self.base_model = base_model

        # Extract multi-level features from SMP models
        if hasattr(base_model.model, 'encoder'):
            # For SMP models, encoder returns features from multiple stages
            self.use_encoder_features = True
            if feature_dims is None:
                # Try to infer feature dimensions
                encoder = base_model.model.encoder
                if hasattr(encoder, 'out_channels'):
                    # Get output channels from each stage
                    out_channels = encoder.out_channels
                    # Use last 3-4 stages
                    feature_dims = out_channels[-3:] if len(out_channels) >= 3 else out_channels
                else:
                    # Default fallback
                    feature_dims = [256, 512, 1024]
        else:
            self.use_encoder_features = False
            if feature_dims is None:
                feature_dims = [256, 512, 1024]  # Default

        # Create loss prediction module
        self.loss_prediction_module = LossPredictionModule(feature_dims, hidden_dim)

    def forward(self, x, return_loss_prediction=False, return_features=False):
        """
        Forward pass with optional loss prediction.

        Args:
            x: Input tensor [B, 3, 512, 512]
            return_loss_prediction: Whether to return predicted loss
            return_features: Whether to return features (for compatibility)

        Returns:
            mask: Segmentation mask [B, 1, 512, 512]
            predicted_loss: Optional [B, 1] if return_loss_prediction=True
            features: Optional features if return_features=True
        """
        # Get base model prediction
        if return_features:
            mask, features = self.base_model(x, return_features=True)
        else:
            mask = self.base_model(x)

        # Extract multi-level features for loss prediction
        if return_loss_prediction:
            if self.use_encoder_features and hasattr(self.base_model.model, 'encoder'):
                # Get encoder features from multiple stages
                encoder_output = self.base_model.model.encoder(x)
                if isinstance(encoder_output, (list, tuple)):
                    # Use last 3-4 feature maps
                    feature_maps = encoder_output[-3:] if len(encoder_output) >= 3 else encoder_output
                else:
                    # Single output, use it
                    feature_maps = [encoder_output]
            else:
                # Fallback: use intermediate decoder features if available
                # For now, create dummy features (this is a limitation)
                batch_size = x.shape[0]
                device = x.device
                # Create dummy multi-level features
                feature_maps = [
                    torch.randn(batch_size, 256, 32, 32, device=device),
                    torch.randn(batch_size, 512, 16, 16, device=device),
                    torch.randn(batch_size, 1024, 8, 8, device=device),
                ]

            # Predict loss using loss prediction module
            predicted_loss = self.loss_prediction_module(feature_maps)  # [B, 1]

            if return_features:
                return mask, predicted_loss, features
            else:
                return mask, predicted_loss
        else:
            if return_features:
                return mask, features
            else:
                return mask


# =============================================================================
# Model Factory Functions
# =============================================================================

def create_model(args, device, checkpoint_path=None):
    """
    Create model for Active Learning training or uncertainty calculation.

    Args:
        args: Arguments containing:
            - mode: Selection strategy (for LUNIT special handling)
            - model_type: Model architecture type
            - task_type: 'segmentation' or 'detection' (optional)
            - dataset_source: 'mdb' or 'vindr' (optional, used to infer task_type)
            - vindr_mask_type: 'detection' or other (optional, used to infer task_type)
        device: PyTorch device
        checkpoint_path: Optional path to load checkpoint

    Returns:
        model: Created model on device
    """
    import os
    from .detection import create_detection_model

    # Determine task type
    task_type = getattr(args, 'task_type', None)
    if task_type is None:
        # Infer from dataset_source and mask_type
        if hasattr(args, 'dataset_source') and args.dataset_source == 'vindr':
            if hasattr(args, 'vindr_mask_type') and args.vindr_mask_type == 'detection':
                task_type = 'detection'
            else:
                task_type = 'segmentation'
        elif hasattr(args, 'dataset_source') and args.dataset_source == 'chestxdet10':
            if getattr(args, 'chestxdet10_mask_type', 'detection') == 'detection':
                task_type = 'detection'
            else:
                task_type = 'segmentation'
        else:
            task_type = 'segmentation'
    
    # Create appropriate model
    if task_type == 'detection':
        return create_detection_model(args, device, checkpoint_path)
    
    # Segmentation models
    if args.mode == 'lunit':
        base_model = SegmentationModel(args.model_type, device).to(device)
        model = SegmentationModelWithLossPrediction(base_model).to(device)
    else:
        model = SegmentationModel(args.model_type, device).to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    return model
