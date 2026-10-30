"""Causal detector integration parity; no downloaded weights or old scores."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from synsteg_audit.detector_formulas import binoculars_score, fast_detectgpt_analytic

SPEC = importlib.util.spec_from_file_location("neural_runner", Path(__file__).resolve().parents[1] / "scripts/rescore_models.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def qwen_job(kind="D5"):
    j = {"kind": kind, "model_id": "Qwen/Qwen2.5-1.5B", "model_revision": "a" * 40,
         "tokenizer_id": "Qwen/Qwen2.5-1.5B", "tokenizer_revision": "a" * 40}
    if kind == "D4":
        j.update(performer_model_id="Qwen/Qwen2.5-1.5B-Instruct", performer_model_revision="b" * 40,
                 performer_tokenizer_id="Qwen/Qwen2.5-1.5B-Instruct", performer_tokenizer_revision="b" * 40)
    return j


def test_precision_must_be_named_and_pins_apply_to_both_models():
    j = r.validate_job(qwen_job("D4"))
    assert j["dtype"] == "float32" and j["max_length"] == 512
    with pytest.raises(ValueError, match="precision_protocol"):
        r.validate_job(dict(j, dtype="bfloat16"))
    b = r.validate_job(dict(j, dtype="bfloat16", precision_protocol="bf16_ablation"))
    assert r.digest(j) != r.digest(b)
    with pytest.raises(ValueError, match="40-character"):
        r.validate_job(dict(j, performer_model_revision="main"))
    with pytest.raises(ValueError, match="requires Qwen"):
        r.validate_job(dict(qwen_job(), model_id="gpt2"))
    with pytest.raises(ValueError, match="cpu_threads"):
        r.validate_job(dict(qwen_job(), cpu_threads=0))


@pytest.mark.parametrize("mask", [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
@pytest.mark.parametrize("chunk", [1, 2, 16])
def test_torch_statistics_match_independent_numpy_formulas(mask, chunk):
    torch = pytest.importorskip("torch")
    gen = np.random.default_rng(36)
    observer = gen.normal(size=(1, 5, 9))
    performer = gen.normal(size=(1, 5, 9))
    ids = np.array([[1, 8, 2, 4, 3]])
    mask = np.array([mask])
    a = r.causal_detector_statistics(torch.tensor(observer), torch.tensor(ids), torch.tensor(mask),
                                     torch.tensor(performer), chunk_positions=chunk)
    b = binoculars_score(observer, performer, ids, mask)
    d5 = fast_detectgpt_analytic(observer, observer, ids, mask)
    assert a["binoculars_ratio"] == pytest.approx(b.score[0], abs=1e-12)
    assert a["observer_fast_detectgpt"] == pytest.approx(d5[0], abs=1e-12)
    assert a["performer_mean_nll"] == pytest.approx(b.performer_nll[0], abs=1e-12)
    assert a["cross_entropy_positions"] == b.n_cross_entropy_positions[0]
    assert a["predicted_tokens"] == b.n_nll_positions[0]
    # Auxiliary PPL is genuinely computed from the same observer likelihood.
    mask_adj = mask[:, :-1].astype(bool) & mask[:, 1:].astype(bool)
    logits = torch.tensor(observer[:, :-1], dtype=torch.float64)
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, 9), torch.tensor(ids[:, 1:]).reshape(-1), reduction="none")
    assert a["observer_prefix_ppl"] == pytest.approx(float(losses[mask_adj.reshape(-1)].mean().exp()), abs=1e-12)


def test_final_position_cross_entropy_and_eos_mask_are_explicit():
    torch = pytest.importorskip("torch")
    obs = np.array([[[2., 0., 1.], [0., 2., 1.], [1., 0., 2.]]])
    perf = np.array([[[0., 1., 2.], [2., 0., 1.], [1., 2., 0.]]])
    ids = np.array([[0, 1, 2]])
    mask = np.ones((1, 3), int)
    for pad in (None, 2):
        a = r.causal_detector_statistics(torch.tensor(obs), torch.tensor(ids), torch.tensor(mask), torch.tensor(perf), pad_token_id=pad)
        b = binoculars_score(obs, perf, ids, mask, pad_token_id=pad)
        assert a["binoculars_ratio"] == pytest.approx(b.score[0], abs=1e-12)
        assert a["cross_entropy_positions"] == (3 if pad is None else 2)
        assert a["predicted_tokens"] == 2


def test_degenerate_d5_is_rejected():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="variance"):
        r.causal_detector_statistics(torch.zeros(1, 2, 3), torch.tensor([[0, 1]]), torch.ones(1, 2))


def test_d4_field_direction_auxiliary_and_tokenization_guard_without_weights():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    s = r.Scorer.__new__(r.Scorer)
    s.torch = torch
    s.device = torch.device("cpu")
    s.job = r.validate_job(qwen_job("D4"))
    class Tok:
        pad_token_id = 2
        def __call__(self, *a, **kw):
            return {"input_ids": torch.tensor([[0, 1, 2]])}
    class Model:
        def __call__(self, *a, **kw):
            return SimpleNamespace(logits=torch.tensor([[[2., 0., 1.], [0., 2., 1.], [1., 0., 2.]]]))
    s.tokenizer = s.performer_tokenizer = Tok()
    s.model = s.performer = Model()
    score, details = s.score("anything")
    assert score == -details["binoculars_ratio"]
    assert details["auxiliary_scores_from_same_forward"] is True
    assert details["observer_prefix_ppl"] == details["performer_prefix_ppl"]
    class DifferentTok(Tok):
        def __call__(self, *a, **kw):
            return {"input_ids": torch.tensor([[1, 1, 2]])}
    s.performer_tokenizer = DifferentTok()
    with pytest.raises(ValueError, match="tokenization differs"):
        s.score("anything")
