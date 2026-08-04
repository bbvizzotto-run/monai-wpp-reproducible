"""Factory for two-dimensional binary classification networks."""

from __future__ import annotations

from typing import Any


class ModelConfigurationError(ValueError):
    """Raised when the requested architecture or options are unsupported."""


def _create_torchvision_resnet(
    name: str,
    *,
    input_channels: int,
    num_classes: int,
) -> Any:
    """Create an ImageNet-pretrained torchvision ResNet adapted to grayscale."""
    if input_channels not in {1, 3}:
        raise ModelConfigurationError(
            "ImageNet-pretrained ResNets currently support 1 or 3 input channels."
        )

    try:
        import torch
        from torch import nn
        from torchvision.models import (
            ResNet18_Weights,
            ResNet50_Weights,
            ResNet101_Weights,
            resnet18,
            resnet50,
            resnet101,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torchvision is required for ImageNet transfer learning. "
            "Install with: pip install -e '.[training]'"
        ) from exc

    builders = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
        "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
    }
    builder, weights = builders[name]
    model = builder(weights=weights)

    if input_channels == 1:
        original_conv = model.conv1
        grayscale_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            grayscale_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
            if original_conv.bias is not None and grayscale_conv.bias is not None:
                grayscale_conv.bias.copy_(original_conv.bias)
        model.conv1 = grayscale_conv

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_model(
    architecture: str,
    *,
    input_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = False,
) -> Any:
    """Create a 2-D classification model.

    ``num_classes=1`` returns one logit for ``BCEWithLogitsLoss``.

    When ``pretrained=True`` is requested for a ResNet architecture, official
    torchvision ImageNet weights are loaded and the first convolution is
    adapted to one-channel input by averaging the RGB kernels. The classifier
    head is replaced with a single-logit output layer.
    """
    if input_channels < 1 or num_classes < 1:
        raise ModelConfigurationError("input_channels and num_classes must be positive.")

    name = architecture.lower().replace("_", "-")
    canonical_resnet_names = {
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
    }
    if pretrained:
        if name not in canonical_resnet_names:
            raise ModelConfigurationError(
                "ImageNet transfer learning is currently enabled only for "
                "resnet18, resnet50, and resnet101."
            )
        return _create_torchvision_resnet(
            canonical_resnet_names[name],
            input_channels=input_channels,
            num_classes=num_classes,
        )

    try:
        from monai.networks.nets import DenseNet121, EfficientNetBN, resnet18, resnet50, resnet101
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MONAI is required to create models. Install with: pip install -e '.[training]'"
        ) from exc

    if name == "resnet18":
        return resnet18(
            spatial_dims=2,
            n_input_channels=input_channels,
            num_classes=num_classes,
            pretrained=False,
        )
    if name == "resnet50":
        return resnet50(
            spatial_dims=2,
            n_input_channels=input_channels,
            num_classes=num_classes,
            pretrained=False,
        )
    if name == "resnet101":
        return resnet101(
            spatial_dims=2,
            n_input_channels=input_channels,
            num_classes=num_classes,
            pretrained=False,
        )
    if name in {"densenet121", "dense-net121"}:
        return DenseNet121(spatial_dims=2, in_channels=input_channels, out_channels=num_classes)
    if name in {"efficientnet-b0", "efficientnetb0"}:
        return EfficientNetBN(
            model_name="efficientnet-b0",
            spatial_dims=2,
            in_channels=input_channels,
            num_classes=num_classes,
            pretrained=False,
        )
    raise ModelConfigurationError(
        "Unsupported architecture. Choose: resnet18, resnet50, resnet101, "
        "densenet121, efficientnet-b0."
    )
