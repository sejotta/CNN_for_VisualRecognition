"""Model definition and transfer-learning utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class LandmarkClassifier(nn.Module):
    """A pretrained timm backbone followed by a small linear classifier.

    ``num_classes=0`` tells timm to replace its original ImageNet classifier
    with an identity, so the backbone returns a pooled feature vector. The new
    linear layer maps that vector to the landmark classes.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        num_features = getattr(self.backbone, "num_features", None)
        if not isinstance(num_features, int) or num_features <= 0:
            # Fallback for unusual timm models. Keep the backbone in eval mode
            # so this dummy pass cannot update BatchNorm running statistics.
            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                features = self.backbone(dummy)
            num_features = int(features.shape[-1])
            self.backbone.train(was_training)

        self.num_features = num_features
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(self.dropout(features))


def freeze_backbone(model: LandmarkClassifier) -> None:
    """Freeze every backbone parameter and keep the new head trainable."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(model: LandmarkClassifier, num_blocks: int) -> int:
    """Unfreeze the last ``num_blocks`` stages of the backbone.

    Most CNN families exposed by timm (MobileNetV3, EfficientNet, ResNet,
    ConvNeXt, ...) store their stages in a ``blocks`` container. The function
    returns the number of stages that were actually unfrozen. If the model has
    no such container, the whole backbone is unfrozen as a safe fallback.
    """
    num_blocks = max(0, int(num_blocks))
    blocks = getattr(model.backbone, "blocks", None)
    block_list = list(blocks.children()) if blocks is not None else []

    if not block_list:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return 0

    num_to_unfreeze = min(num_blocks, len(block_list))
    if num_to_unfreeze:
        for block in block_list[-num_to_unfreeze:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return num_to_unfreeze


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep BatchNorm statistics frozen for parts of the network that are frozen.

    ``model.train()`` puts every BatchNorm layer back into training mode, which
    would update running statistics even in a frozen backbone. After calling
    ``model.train()``, this helper restores eval mode for BatchNorm modules that
    have no trainable affine parameters.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
