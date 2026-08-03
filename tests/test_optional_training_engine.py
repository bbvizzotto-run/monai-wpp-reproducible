from pathlib import Path

import pytest


def test_training_engine_creates_checkpoint_and_predictions(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, Dataset

    from monai_wpp.training import fit_binary_classifier, predict_binary_classifier

    class TinyDataset(Dataset):
        def __init__(self) -> None:
            self.labels = [0, 1, 0, 1]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int):
            label = self.labels[index]
            image = torch.full((1, 8, 8), float(label), dtype=torch.float32)
            return {
                "image": image,
                "label": label,
                "case_id": f"case_{index}",
                "patient_id": f"patient_{index}",
            }

    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 1))
    checkpoint = tmp_path / "best_model.pt"
    history = tmp_path / "history.csv"

    result = fit_binary_classifier(
        model,
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=1e-2,
        patience=2,
        checkpoint_path=checkpoint,
        history_path=history,
    )

    predictions = predict_binary_classifier(model, loader, torch.device("cpu"))

    assert checkpoint.is_file()
    assert history.is_file()
    assert not result.empty
    assert predictions["case_id"].is_unique
    assert len(predictions) == 4
    assert predictions["probability"].between(0.0, 1.0).all()
