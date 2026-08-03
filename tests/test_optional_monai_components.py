import pytest


def test_model_factory_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    from monai_wpp.models import create_model

    model = create_model("resnet18")
    output = model(torch.zeros(2, 1, 224, 224))
    assert tuple(output.shape) == (2, 1)


def test_training_transform_preserves_scalar_label() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("monai")
    from monai_wpp.data.transforms import build_classification_transforms

    transform = build_classification_transforms(image_size=(64, 64), training=True)
    result = transform({"image": np.zeros((1, 80, 120), dtype=np.float32), "label": 1})
    assert tuple(result["image"].shape) == (1, 64, 64)
    assert int(result["label"]) == 1
