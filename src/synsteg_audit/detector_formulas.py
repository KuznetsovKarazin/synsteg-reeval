"""Independent float64 NumPy scoring kernels; no model inference or thresholds.

Inputs are unshifted causal logits [batch, sequence, vocabulary], token IDs and
attention masks [batch, sequence]. Both models must use identical vocabulary
indices. See docs/detector_corrections.md for pinned primary implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _inputs(logits, input_ids, attention_mask):
    x = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(input_ids)
    mask = np.asarray(attention_mask)
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] < 2 or x.shape[2] < 2:
        raise ValueError("logits require nonempty batch, at least two positions and two vocabulary entries")
    if ids.shape != x.shape[:2] or mask.shape != ids.shape:
        raise ValueError("token IDs and attention mask must match batch and sequence dimensions")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("token IDs must be integers")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("attention mask must be binary")
    mask = mask.astype(bool)
    if np.any(ids[mask] < 0) or np.any(ids[mask] >= x.shape[-1]):
        raise ValueError("observed token ID outside vocabulary")
    if not np.all(np.isfinite(x[mask])):
        raise ValueError("observed logits must be finite")
    # Prevent invalid pad values (including NaN logits) from affecting reductions.
    x = np.where(mask[..., None], x, 0.0)
    ids = np.where(mask, ids, 0)
    adjacent = mask[:, :-1] & mask[:, 1:]
    if np.any(adjacent.sum(axis=1) == 0):
        raise ValueError("each sample needs at least one observed next-token transition")
    return x, ids, mask, adjacent


def _log_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _paired_inputs(first, second, input_ids, attention_mask):
    x, ids, mask, adjacent = _inputs(first, input_ids, attention_mask)
    y, _, _, _ = _inputs(second, input_ids, attention_mask)
    if x.shape != y.shape:
        raise ValueError("models must have identical sequence and aligned vocabulary shapes")
    return x, y, ids, mask, adjacent


def _mean(values, mask):
    n = mask.sum(axis=1)
    if np.any(n == 0):
        raise ValueError("no valid positions in reduction")
    return np.where(mask, values, 0.0).sum(axis=1) / n


@dataclass(frozen=True)
class BinocularsResult:
    score: np.ndarray  # Lower is more AI-like, as in the upstream implementation.
    performer_nll: np.ndarray
    observer_performer_cross_entropy: np.ndarray
    n_nll_positions: np.ndarray
    n_cross_entropy_positions: np.ndarray

    @property
    def ai_score(self) -> np.ndarray:
        """Higher-is-more-AI orientation for common ROC interfaces."""
        return -self.score


def binoculars_score(
    observer_logits,
    performer_logits,
    input_ids,
    attention_mask,
    *,
    cross_entropy_positions: Literal["all", "next_token"] = "all",
    pad_token_id: int | None = None,
) -> BinocularsResult:
    """Performer NLL / H(observer, performer), in log-probability units.

    Default cross-entropy uses every observed input position, including the final
    one, matching upstream's position convention. Optional ``next_token`` is a
    separately named ablation, not the published implementation. Padding is
    excluded by attention mask; passing ``pad_token_id`` additionally excludes
    those IDs from cross-entropy to reproduce upstream's ID-based mask.
    """
    lo, lp, ids, mask, adjacent = _paired_inputs(
        observer_logits, performer_logits, input_ids, attention_mask)
    log_o, log_p = _log_softmax(lo), _log_softmax(lp)
    nll_by_position = -np.take_along_axis(log_p[:, :-1], ids[:, 1:, None], axis=-1)[..., 0]
    nll = _mean(nll_by_position, adjacent)
    cross = -(np.exp(log_o) * log_p).sum(axis=-1)
    if cross_entropy_positions == "all":
        cross_mask = mask.copy()
        if pad_token_id is not None:
            cross_mask &= ids != pad_token_id
    elif cross_entropy_positions == "next_token":
        cross = cross[:, :-1]
        cross_mask = adjacent.copy()
        if pad_token_id is not None:
            cross_mask &= ids[:, :-1] != pad_token_id
    else:
        raise ValueError("cross_entropy_positions must be 'all' or 'next_token'")
    ce = _mean(cross, cross_mask)
    if np.any(ce <= 0) or not np.all(np.isfinite(ce)):
        raise ValueError("cross-entropy denominator is not finite and positive")
    return BinocularsResult(nll / ce, nll, ce, adjacent.sum(axis=1), cross_mask.sum(axis=1))


def fast_detectgpt_analytic(reference_logits, scoring_logits, input_ids, attention_mask) -> np.ndarray:
    """Sum-centered conditional log likelihood / sqrt(sum of variances).

    Returns one raw discrepancy per sample. Token positions are shifted once,
    before reduction, and masked by valid input/target adjacency. No probability
    calibration or transfer threshold is implied. Degenerate zero variance is
    rejected rather than silently producing NaN or assigning a detector label.
    """
    ref, score, ids, _, adjacent = _paired_inputs(
        reference_logits, scoring_logits, input_ids, attention_mask)
    lr = _log_softmax(ref[:, :-1])
    ls = _log_softmax(score[:, :-1])
    pr = np.exp(lr)
    observed = np.take_along_axis(ls, ids[:, 1:, None], axis=-1)[..., 0]
    mean = (pr * ls).sum(axis=-1)
    # Centered second moment is algebraically equivalent to E[L²] - E[L]²
    # but avoids cancellation of two close floating-point values.
    var = (pr * (ls - mean[..., None]) ** 2).sum(axis=-1)
    numerator = np.where(adjacent, observed - mean, 0.0).sum(axis=1)
    variance = np.where(adjacent, var, 0.0).sum(axis=1)
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ValueError("sampling discrepancy requires finite positive conditional variance")
    return numerator / np.sqrt(variance)
