"""Factory for two-dimensional binary classification networks."""

from __future__ import annotations

from typing import Any


class ModelConfigurationError(ValueError):
    """Raised when the requested architecture or options are unsupported."""


def create_model(
    architecture: str,
    *,
    input_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = False,
) -> Any:
    """Create a MONAI 2-D classification model.

    ``num_classes=1`` returns one logit for use with ``BCEWithLogitsLoss``.
    Pretrained weights are deliberately disabled until the definitive transfer-
    learning source and channel adaptation strategy are documented.
    """
    if input_channels < 1 or num_classes < 1:
        raise ModelConfigurationError("input_channels and num_classes must be positive.")
    if pretrained:
        raise ModelConfigurationError(
            "Pretrained weights are not enabled yet. Their source and channel adaptation "
            "must be fixed before definitive experiments."
        )

    try:
        from monai.networks.nets import DenseNet121, EfficientNetBN, resnet18, resnet50, resnet101
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MONAI is required to create models. Install with: pip install -e '.[training]'"
        ) from exc

    name = architecture.lower().replace("_", "-")
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
