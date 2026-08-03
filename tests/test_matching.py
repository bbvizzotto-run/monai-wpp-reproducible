from pathlib import Path

import pandas as pd

from monai_wpp.data.matching import suggest_pair_candidates
from monai_wpp.utils.synthetic import generate_synthetic_dataset


def test_pair_suggestion_ranks_true_counterpart_first(tmp_path: Path) -> None:
    manifest_path = generate_synthetic_dataset(tmp_path, n_cases=8, seed=17)
    manifest = pd.read_csv(manifest_path)
    candidates = suggest_pair_candidates(tmp_path / "original", tmp_path / "whatsapp", top_k=1)
    expected = {
        str((tmp_path / row.original_path)): str((tmp_path / row.whatsapp_path))
        for row in manifest.itertuples()
    }
    observed = dict(zip(candidates["original_file"], candidates["whatsapp_candidate"], strict=True))
    assert observed == expected
    assert (candidates["status"] == "review").all()
