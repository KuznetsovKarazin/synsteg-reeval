"""Independent checkpoint-reference formula tests, no model loading."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from synsteg_audit.detector_formulas import binoculars_score, fast_detectgpt_analytic

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("d4_check", ROOT / "scripts/check_d4_checkpoint.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize("mask", [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
@pytest.mark.parametrize("pad", [None, 2])
def test_independent_reference_matches_audited_formulas(mask, pad):
    rng = np.random.default_rng(827)
    observer, performer = rng.normal(size=(1, 5, 7)), rng.normal(size=(1, 5, 7))
    ids, mask = np.array([[1, 4, 2, 5, 2]]), np.array([mask])
    actual = check.independent_statistics(observer, performer, ids, mask, pad)
    d4 = binoculars_score(observer, performer, ids, mask, pad_token_id=pad)
    d5 = fast_detectgpt_analytic(observer, observer, ids, mask)
    assert actual["binoculars_ratio"] == pytest.approx(d4.score[0], abs=1e-12)
    assert actual["observer_fast_detectgpt"] == pytest.approx(d5[0], abs=1e-12)
    assert actual["performer_mean_nll"] == pytest.approx(d4.performer_nll[0], abs=1e-12)
    assert actual["predicted_tokens"] == d4.n_nll_positions[0]
    assert actual["cross_entropy_positions"] == d4.n_cross_entropy_positions[0]


def test_final_position_only_changes_cross_entropy():
    observer = np.array([[[2., 0., 1.], [0., 2., 1.], [1., 0., 2.]]])
    performer = observer.copy()
    ids, mask = np.array([[0, 1, 2]]), np.ones((1, 3), int)
    before = check.independent_statistics(observer, performer, ids, mask)
    performer[0, -1] = [9., -4., 0.]
    after = check.independent_statistics(observer, performer, ids, mask)
    assert before["performer_prefix_ppl"] == after["performer_prefix_ppl"]
    assert before["observer_fast_detectgpt"] == after["observer_fast_detectgpt"]
    assert before["binoculars_ratio"] != after["binoculars_ratio"]
    assert after["cross_entropy_positions"] == 3 and after["predicted_tokens"] == 2


def test_uniform_roundoff_cannot_create_variance():
    logits = np.zeros((1, 3, 7))
    with pytest.raises(ValueError, match="variance"):
        check.independent_statistics(logits, logits, np.array([[0, 1, 2]]), np.ones((1, 3), int))


def test_plan_and_exclusive_gate_do_not_access_corpus(tmp_path):
    target = tmp_path / "absent"
    command = [sys.executable, str(ROOT / "scripts/check_d4_checkpoint.py"),
               "--corpus", str(tmp_path / "missing.jsonl"), "--output", str(target)]
    result = subprocess.run(command + ["--plan-only"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["requires_exclusive_model_workload"] is True
    assert not target.exists()
    denied = subprocess.run(command, capture_output=True, text=True)
    assert denied.returncode != 0 and "--exclusive-confirmed" in denied.stderr
    assert "FileNotFoundError" not in denied.stderr
