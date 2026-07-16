#!/usr/bin/env python3
"""
Detection models for object detection tasks.

This module contains detection models compatible with the active learning framework,
including Faster R-CNN and RetinaNet implementations.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          retinanet_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead


class DetectionModel(nn.Module):
    """
    Detection model wrapper for object detection tasks.

    Supports Faster R-CNN and RetinaNet architectures.
    """

    def __init__(self, model_type: str = 'faster_rcnn', num_classes: int = 2, device: str = 'cuda'):
        """
        Args:
            model_type: 'faster_rcnn' or 'retinanet'
            num_classes: Number of classes (including background), default 2 (background + lesion)
            device: Device to run model on
        """
        super().__init__()

        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device

        if model_type == 'faster_rcnn':
            # Load pre-trained Faster R-CNN
            self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
            # Replace the classifier head with num_classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        elif model_type == 'retinanet':
            # Load pre-trained RetinaNet
            self.model = retinanet_resnet50_fpn(weights='DEFAULT')
            # Replace the classification head
            in_features = self.model.head.classification_head.conv[0].in_channels
            num_anchors = self.model.head.classification_head.num_anchors
            self.model.head.classification_head.num_classes = num_classes
            # Recreate classification head
            cls_logits = nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(cls_logits.weight, std=0.01)
            nn.init.constant_(cls_logits.bias, -4.595)  # -log((1-0.01)/0.01)
            self.model.head.classification_head.cls_logits = cls_logits
        else:
            raise ValueError(f"Unknown detection model type: {model_type}. Choose from: faster_rcnn, retinanet")

        self.model = self.model.to(device)

    def forward(self, images, targets=None, return_features=False):
        """
        Forward pass for detection model.

        Args:
            images: List of images [C, H, W] or tensor [B, C, H, W]
            targets: Optional list of target dicts with 'boxes' and 'labels'
            return_features: Whether to return features (for compatibility)

        Returns:
            During training: dict with 'loss' keys
            During inference: list of dicts with 'boxes', 'labels', 'scores'
            If return_features: also returns features
        """
        # Convert to list format if needed (torchvision detection models expect list)
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                # Batch tensor [B, C, H, W] -> list of [C, H, W]
                images = [img for img in images]
            else:
                images = [images]

        if self.training and targets is not None:
            # Training mode: return losses
            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())
            return {'loss': loss, **loss_dict}
        else:
            # Inference mode: return predictions
            predictions = self.model(images)

            if return_features:
                # Extract features from backbone (for compatibility with uncertainty calculation)
                # Note: This is a simplified feature extraction
                # In practice, you might want to extract from specific layers
                with torch.no_grad():
                    # Use backbone to extract features
                    if hasattr(self.model, 'backbone'):
                        # Get features from backbone
                        features_list = []
                        for img in images:
                            # Extract features (simplified - actual implementation depends on model)
                            if hasattr(self.model.backbone, 'forward'):
                                feat = self.model.backbone(img.unsqueeze(0))
                                # Global average pooling
                                if isinstance(feat, dict):
                                    # FPN returns dict, use P5 (highest resolution)
                                    feat = feat.get('3', list(feat.values())[0])
                                if feat.dim() > 2:
                                    feat = torch.mean(feat, dim=(2, 3))  # Global average pooling
                                features_list.append(feat.squeeze())
                            else:
                                # Fallback: create dummy features
                                features_list.append(torch.zeros(256, device=img.device))
                        features = torch.stack(features_list)  # [B, D]
                    else:
                        # Fallback: create dummy features
                        batch_size = len(images)
                        features = torch.zeros(batch_size, 256, device=images[0].device)

                return predictions, features
            else:
                return predictions

    def predict(self, images):
        """Convenience method for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(images)


def create_detection_model(args, device, checkpoint_path=None):
    """
    Create detection model for Active Learning.

    Args:
        args: Arguments containing:
            - model_type: 'faster_rcnn' or 'retinanet'
            - num_classes: Number of classes (default 2 for binary detection)
        device: PyTorch device
        checkpoint_path: Optional path to load checkpoint

    Returns:
        model: Created detection model on device
    """
    import os

    model_type = getattr(args, 'detection_model_type', 'faster_rcnn')
    num_classes = getattr(args, 'num_classes', 2)  # background + lesion

    model = DetectionModel(model_type=model_type, num_classes=num_classes, device=device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"📥 Loaded detection model checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Failed to load detection checkpoint: {e}")

    return model
