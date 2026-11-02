"""Automatic captioning metrics and BLEU/hallucination analysis.

Contents
--------
* :func:`corpus_bleu` / :func:`sentence_bleu` -- SacreBLEU with the 13a
  tokenizer, reported as percentages (0-100).
* :func:`cider_d` -- **a from-scratch CIDEr-D implementation** (Vedantam et al.,
  CVPR 2015), so the project needs no Java and no external scorer install.
* :func:`evaluate_predictions` -- BLEU + CIDEr-D + CHAIR in one call.
* :func:`bootstrap_ci`, :func:`paired_bootstrap_difference`,
  :func:`spearman_correlation` -- uncertainty and association statistics.
* :func:`bleu_hallucination_analysis` -- the research-question analysis: is
  higher caption quality associated with fewer hallucinated objects?

Every numeric value returned by the public functions is a native Python
``float``/``int``/``bool``/``None`` (never ``np.float32`` or ``np.int64``), so
``json.dump`` works on the result without a custom encoder.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sacrebleu
from scipy import stats

from .hallucination import score_hallucination

__all__ = [
    "bleu_hallucination_analysis",
    "bootstrap_ci",
    "cider_d",
    "corpus_bleu",
    "evaluate_predictions",
    "paired_bootstrap_difference",
    "sentence_bleu",
    "spearman_correlation",
]

#: Max n-gram order used by CIDEr-D.
_CIDER_MAX_N = 4
#: CIDEr-D Gaussian length-penalty standard deviation (Vedantam et al., 2015).
_CIDER_SIGMA = 6.0
#: Simple metric tokenizer: lowercase, strip punctuation, split on whitespace.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _native(value: Any) -> Any:
    """Convert a numpy scalar to the equivalent native Python object."""
    if isinstance(value, np.generic):
        return value.item()
    return value


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #


def _as_reference_streams(
    references: Sequence[Sequence[str]], n_items: int
) -> list[list[str]]:
    """Transpose per-item reference lists into SacreBLEU's per-annotator streams.

    SacreBLEU expects ``references[annotator][item]``, while callers naturally
    hold ``references[item][annotator]``.  Shorter reference lists are padded
    with empty strings, which SacreBLEU treats as "no reference from this
    annotator".
    """
    if len(references) != n_items:
        raise ValueError(
            f"expected {n_items} reference lists, got {len(references)}"
        )
    width = max((len(refs) for refs in references), default=0)
    streams: list[list[str]] = []
    for index in range(width):
        streams.append(
            [
                str(references[item][index])
                if index < len(references[item])
                else ""
                for item in range(n_items)
            ]
        )
    return streams


def corpus_bleu(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    """Corpus-level BLEU-1..4 with SacreBLEU's ``13a`` tokenizer.

    BLEU-k here is the standard cumulative BLEU over 1..k grams (what COCO
    reports as BLEU-1..4), not the raw k-gram precision.

    Args:
        predictions: Items with a ``caption`` field.
        references: ``references[i]`` is the list of reference captions for
            ``predictions[i]``.

    Returns:
        Dict with ``bleu1``..``bleu4`` as percentages (0-100), plus the BLEU-4
        ``brevity_penalty``, ``sys_len`` and ``ref_len`` for diagnostics.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    hypotheses = [str(item.get("caption") or "") for item in predictions]
    streams = _as_reference_streams(references, len(predictions))

    scores: dict[str, float | int] = {}
    last = None
    for order in range(1, 5):
        metric = sacrebleu.BLEU(tokenize="13a", max_ngram_order=order)
        last = metric.corpus_score(hypotheses, streams)
        # sacrebleu can return 100.00000000000004 from float rounding; clamp so
        # the documented 0-100 contract holds exactly.
        scores[f"bleu{order}"] = min(100.0, max(0.0, float(last.score)))

    assert last is not None
    scores["brevity_penalty"] = float(last.bp)
    scores["sys_len"] = int(last.sys_len)
    scores["ref_len"] = int(last.ref_len)
    return scores


def sentence_bleu(
    hypothesis: str,
    references: Sequence[str],
    max_order: int = 4,
) -> float:
    """Sentence BLEU (percentage 0-100) with exponential smoothing.

    ``effective_order=True`` stops at the highest order actually observed, which
    is what makes short captions ("a dog") score sensibly instead of 0.

    Args:
        hypothesis: Generated caption.
        references: Reference captions for the same image.
        max_order: Maximum n-gram order (4 for BLEU-4, 2 for BLEU-2).

    Returns:
        BLEU in 0-100, or ``0.0`` when ``references`` is empty.
    """
    refs = [str(ref) for ref in references if ref is not None]
    if not refs:
        return 0.0
    metric = sacrebleu.BLEU(
        tokenize="13a",
        smooth_method="exp",
        effective_order=True,
        max_ngram_order=int(max_order),
    )
    score = float(metric.sentence_score(str(hypothesis or ""), refs).score)
    # Clamp the float-rounding overshoot (e.g. 100.00000000000004) so the
    # documented 0-100 contract holds exactly.
    return min(100.0, max(0.0, score))


# --------------------------------------------------------------------------- #
# CIDEr-D (from scratch)
# --------------------------------------------------------------------------- #


def _cider_tokens(text: str) -> list[str]:
    """Lowercase, punctuation-stripped whitespace tokens."""
    return _TOKEN_RE.findall((text or "").lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Count the n-grams of ``tokens``."""
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def cider_d(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[str]],
) -> float:
    """CIDEr-D score (0-10) computed from scratch.

    This is a dependency-free reimplementation of CIDEr-D (Vedantam, Lawrence
    Zitnick & Parikh, CVPR 2015), following the reference formulation:

    1. Count n-grams (n = 1..4) for the hypothesis and every reference.
    2. Weight each n-gram by TF-IDF over the reference set::

           idf(omega) = log((N + 1) / (df(omega) + 1)) + 1

       where ``df`` is the number of *images* whose reference captions contain
       ``omega`` (Vedantam et al., Eq. 4).  The ``+1`` smoothing keeps every
       weight finite and positive, including n-grams present in all references.
    3. Score each (hypothesis, reference) pair with a clipped cosine similarity

           min(vec_hyp, vec_ref) . vec_ref
           ------------------------------
              ||vec_hyp|| * ||vec_ref||

       multiplied by the Gaussian length penalty
       ``exp(-(len_hyp - len_ref)**2 / (2 * sigma**2))`` with ``sigma = 6``.
    4. Average over the ``k = 1..4`` orders and over all references of the
       image, then multiply by 10.

    The score is the mean over images, so it is deterministic and independent of
    prediction order.  Empty captions contribute 0 rather than raising.

    Args:
        predictions: Items with a ``caption`` field.
        references: ``references[i]`` is the list of reference captions for
            ``predictions[i]``.

    Returns:
        Mean CIDEr-D in ``[0, 10]`` (0.0 for empty input).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return 0.0

    hypotheses = [_cider_tokens(str(item.get("caption") or "")) for item in predictions]
    ref_tokens = [
        [_cider_tokens(str(ref)) for ref in (refs or [])] for refs in references
    ]

    # Document frequency over reference captions (not over images).
    # Image-level document frequency (Vedantam et al., Eq. 4): an n-gram's df
    # counts how many *images* have it in at least one reference caption.  With
    # caption-level counts, df could exceed N and every IDF would collapse to
    # zero for small corpora (a perfect caption would score 0).
    document_frequency: Counter[tuple[str, ...]] = Counter()
    for refs in ref_tokens:
        image_ngrams: set[tuple[str, ...]] = set()
        for ref in refs:
            for n in range(1, _CIDER_MAX_N + 1):
                image_ngrams.update(_ngram_counts(ref, n))
        document_frequency.update(image_ngrams)

    n_documents = sum(1 for refs in ref_tokens if refs)
    if n_documents == 0:
        return 0.0

    # Additive smoothing keeps IDF finite and strictly positive: an n-gram seen
    # in every image still receives a small, non-zero weight.
    def idf(ngram: tuple[str, ...]) -> float:
        df = float(document_frequency.get(ngram, 0))
        return math.log((float(n_documents) + 1.0) / (df + 1.0)) + 1.0

    def tfidf_vector(
        tokens: Sequence[str], n: int
    ) -> tuple[dict[tuple[str, ...], float], float]:
        """Return the TF-IDF vector and its L2 norm for order ``n``."""
        vector: dict[tuple[str, ...], float] = {}
        for ngram, count in _ngram_counts(tokens, n).items():
            weight = float(count) * idf(ngram)
            if weight > 0.0:
                vector[ngram] = weight
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        return vector, norm

    image_scores: list[float] = []
    for hypothesis, refs in zip(hypotheses, ref_tokens):
        if not refs:
            image_scores.append(0.0)
            continue

        # Hypothesis vectors/norms are reference-independent; compute once.
        hyp_vectors = [tfidf_vector(hypothesis, n) for n in range(1, _CIDER_MAX_N + 1)]

        score = np.zeros(_CIDER_MAX_N, dtype=np.float64)
        for ref in refs:
            delta = float(len(hypothesis) - len(ref))
            penalty = math.exp(-(delta * delta) / (2.0 * _CIDER_SIGMA**2))
            for index in range(_CIDER_MAX_N):
                hyp_vec, hyp_norm = hyp_vectors[index]
                ref_vec, ref_norm = tfidf_vector(ref, index + 1)
                if hyp_norm == 0.0 or ref_norm == 0.0 or not hyp_vec or not ref_vec:
                    continue
                overlap = 0.0
                for ngram, hyp_weight in hyp_vec.items():
                    ref_weight = ref_vec.get(ngram)
                    if ref_weight is None:
                        continue
                    overlap += min(hyp_weight, ref_weight) * ref_weight
                score[index] += overlap / (hyp_norm * ref_norm) * penalty

        per_ref = score / float(len(refs))
        image_scores.append(float(np.mean(per_ref)) * 10.0)

    mean_score = float(np.mean(image_scores)) if image_scores else 0.0
    return float(min(10.0, max(0.0, mean_score)))


# --------------------------------------------------------------------------- #
# Bootstrap / correlation statistics
# --------------------------------------------------------------------------- #


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    """Convert any 1-D array-like to a float64 numpy array."""
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    return np.asarray(array, dtype=np.float64).ravel()


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Resamples ``values`` with replacement ``n_boot`` times and returns the
    ``alpha/2`` and ``1 - alpha/2`` quantiles of the resampled means.

    Args:
        values: 1-D array-like of per-image statistics.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level (0.05 -> 95% CI).
        seed: Seed for ``numpy.random.default_rng`` (reproducibility).

    Returns:
        ``(low, high)``.  Returns ``(nan, nan)`` for empty input and collapses to
        ``(value, value)`` for a single observation.
    """
    array = _as_float_array(values)
    if array.size == 0:
        return (float("nan"), float("nan"))
    n_boot = max(1, int(n_boot))
    rng = np.random.default_rng(seed)
    if array.size == 1:
        value = float(array[0])
        return (value, value)
    draws = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[draws].mean(axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (low, high)


def paired_bootstrap_difference(
    a: Iterable[float],
    b: Iterable[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap for ``mean(a - b)``, resampling images with replacement.

    Used to compare two models on the same images (per-image hallucination
    indicators, BLEU, hallucianted-object counts), where pairing removes
    between-image variance.

    Args:
        a: First model's per-image values.
        b: Second model's per-image values (same image order as ``a``).
        n_boot: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``{"mean_diff", "ci_low", "ci_high", "p_two_sided"}`` where the p-value
        is ``2 * min(fraction(diff <= 0), fraction(diff >= 0))`` clipped to
        ``[0, 1]`` over the bootstrap distribution of the mean difference.

    Raises:
        ValueError: If the two arrays have different lengths.
    """
    a_arr = _as_float_array(a)
    b_arr = _as_float_array(b)
    if a_arr.size != b_arr.size:
        raise ValueError(
            f"paired arrays must have equal length ({a_arr.size} != {b_arr.size})"
        )
    if a_arr.size == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_two_sided": 1.0}

    diff = a_arr - b_arr
    n_boot = max(1, int(n_boot))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_means = diff[draws].mean(axis=1)

    p_left = float(np.mean(boot_means <= 0.0))
    p_right = float(np.mean(boot_means >= 0.0))
    p_value = min(1.0, max(0.0, 2.0 * min(p_left, p_right)))

    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.quantile(boot_means, 0.025)),
        "ci_high": float(np.quantile(boot_means, 0.975)),
        "p_two_sided": float(p_value),
    }


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    """Spearman rank correlation with graceful handling of edge cases.

    Args:
        x: First variable.
        y: Second variable (same length as ``x``).

    Returns:
        ``{"rho", "p_value", "n"}``.  ``rho`` and ``p_value`` are ``None`` when
        the correlation is undefined: fewer than three paired points, unequal
        lengths, non-finite input, or a constant array (scipy would return
        ``nan`` with a ``ConstantInputWarning``; we avoid emitting that warning
        and report ``None`` instead so the value stays JSON-serializable).
    """
    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)
    n = int(min(x_arr.size, y_arr.size))
    if x_arr.size != y_arr.size or n < 3:
        return {"rho": None, "p_value": None, "n": n}
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        return {"rho": None, "p_value": None, "n": n}
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return {"rho": None, "p_value": None, "n": n}

    result = stats.spearmanr(x_arr, y_arr)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(rho):
        return {"rho": None, "p_value": None, "n": n}
    if not math.isfinite(p_value):
        p_value = float("nan")  # pragma: no cover - scipy edge case
    return {"rho": rho, "p_value": p_value, "n": n}


def _quartile_labels(values: np.ndarray) -> np.ndarray:
    """Assign rank-based quartile labels 0..3, robust to ties and small n."""
    n = values.size
    order = np.argsort(values, kind="stable")
    labels = np.empty(n, dtype=np.int64)
    for rank, index in enumerate(order):
        labels[index] = min(3, int(4 * rank / n))
    return labels


# --------------------------------------------------------------------------- #
# End-to-end evaluation
# --------------------------------------------------------------------------- #


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    gt_objects: Mapping[Any, Sequence[str]],
) -> dict[str, Any]:
    """Compute BLEU, CIDEr-D and CHAIR for one model's predictions.

    Args:
        predictions: Items with ``image_id``, ``caption`` and ``references``.
        gt_objects: Mapping from image id (str) to COCO class names.

    Returns:
        Dict with ``n``, ``bleu1``..``bleu4``, ``cider_d``, ``chair_s``,
        ``chair_i``, ``object_precision``, ``object_recall``, ``object_f1``,
        ``n_mentions``, ``n_hallucinated`` and ``per_image``.  Each per-image
        record is the CHAIR record augmented with ``sentence_bleu4`` and
        ``sentence_bleu2`` (``bleu_bucket`` is assigned downstream by
        :func:`bleu_hallucination_analysis`).  Every value is JSON-safe.

    Raises:
        ValueError: If ``predictions`` is empty.
    """
    if not predictions:
        raise ValueError("predictions must not be empty")

    references = [list(item.get("references") or []) for item in predictions]
    bleu = corpus_bleu(predictions, references)
    chair = score_hallucination(predictions, gt_objects)
    cider = cider_d(predictions, references)

    per_image = chair["per_image"]
    for record in per_image:
        record["sentence_bleu4"] = sentence_bleu(
            record["caption"], record["references"], max_order=4
        )
        record["sentence_bleu2"] = sentence_bleu(
            record["caption"], record["references"], max_order=2
        )
        record.setdefault("bleu_bucket", None)

    return {
        "n": int(len(predictions)),
        "bleu1": float(bleu["bleu1"]),
        "bleu2": float(bleu["bleu2"]),
        "bleu3": float(bleu["bleu3"]),
        "bleu4": float(bleu["bleu4"]),
        "brevity_penalty": float(bleu["brevity_penalty"]),
        "cider_d": float(cider),
        "chair_s": float(chair["chair_s"]),
        "chair_i": float(chair["chair_i"]),
        "object_precision": float(chair["object_precision"]),
        "object_recall": float(chair["object_recall"]),
        "object_f1": float(chair["object_f1"]),
        "n_mentions": int(chair["n_mentions"]),
        "n_hallucinated": int(chair["n_hallucinated"]),
        "per_image": per_image,
    }


def bleu_hallucination_analysis(
    per_image: Sequence[Mapping[str, Any]],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Test whether higher caption BLEU goes with fewer hallucinated objects.

    Computes:

    * Spearman rho between ``sentence_bleu4`` and ``n_hallucinated``.
    * Spearman rho between ``sentence_bleu2`` and ``has_hallucination``.
    * A BLEU-4 quartile table (Q1..Q4) with the mean hallucination rate, its
      bootstrap CI, and the BLEU range of each bucket.
    * The overall mean hallucination rate and its bootstrap CI.

    Args:
        per_image: Records from :func:`evaluate_predictions`, each containing
            ``sentence_bleu4``, ``sentence_bleu2``, ``n_hallucinated`` and
            ``has_hallucination``.
        n_boot: Bootstrap resamples for the confidence intervals.
        seed: RNG seed.

    Returns:
        A JSON-serializable dict; all numbers are native Python types.
    """
    if not per_image:
        raise ValueError("per_image must not be empty")

    bleu4 = np.asarray([float(r["sentence_bleu4"]) for r in per_image], dtype=np.float64)
    bleu2 = np.asarray([float(r["sentence_bleu2"]) for r in per_image], dtype=np.float64)
    hallucinated = np.asarray(
        [float(r["n_hallucinated"]) for r in per_image], dtype=np.float64
    )
    indicators = np.asarray(
        [1.0 if r["has_hallucination"] else 0.0 for r in per_image], dtype=np.float64
    )

    labels = _quartile_labels(bleu4)
    quartiles: list[dict[str, Any]] = []
    for label in range(4):
        mask = labels == label
        if not mask.any():
            continue
        mask_values = indicators[mask]
        low, high = bootstrap_ci(mask_values, n_boot=n_boot, seed=seed + 1 + label)
        quartiles.append(
            {
                "bucket": f"Q{label + 1}",
                "n": int(mask.sum()),
                "bleu4_min": float(bleu4[mask].min()),
                "bleu4_max": float(bleu4[mask].max()),
                "bleu4_mean": float(bleu4[mask].mean()),
                "mean_hallucinated_objects": float(hallucinated[mask].mean()),
                "hallucination_rate": float(mask_values.mean()),
                "hallucination_rate_ci_low": float(low),
                "hallucination_rate_ci_high": float(high),
            }
        )

    rate = float(indicators.mean())
    rate_low, rate_high = bootstrap_ci(indicators, n_boot=n_boot, seed=seed)
    mean_hallucinated = float(hallucinated.mean())
    halluc_low, halluc_high = bootstrap_ci(hallucinated, n_boot=n_boot, seed=seed + 101)

    return {
        "n": int(len(per_image)),
        "spearman_bleu4_vs_n_hallucinated": spearman_correlation(bleu4, hallucinated),
        "spearman_bleu2_vs_has_hallucination": spearman_correlation(bleu2, indicators),
        "hallucination_rate": rate,
        "hallucination_rate_ci": [float(rate_low), float(rate_high)],
        "mean_hallucinated_objects": mean_hallucinated,
        "mean_hallucinated_objects_ci": [float(halluc_low), float(halluc_high)],
        "bleu4_quartiles": quartiles,
    }


def aggregate_summary(metrics_by_model: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact model-level rows (percentages where the report expects them).

    CHAIR and object metrics are multiplied by 100 here to match how the report
    and comparison table present them; BLEU and CIDEr-D are already
    percentages / 0-10 in :func:`evaluate_predictions`.
    """
    rows: list[dict[str, Any]] = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "n": int(metrics["n"]),
                "bleu1": float(metrics["bleu1"]),
                "bleu2": float(metrics["bleu2"]),
                "bleu3": float(metrics["bleu3"]),
                "bleu4": float(metrics["bleu4"]),
                "cider_d": float(metrics["cider_d"]),
                "chair_s": float(metrics["chair_s"]) * 100.0,
                "chair_i": float(metrics["chair_i"]) * 100.0,
                "object_precision": float(metrics["object_precision"]) * 100.0,
                "object_recall": float(metrics["object_recall"]) * 100.0,
                "object_f1": float(metrics["object_f1"]) * 100.0,
                "n_mentions": int(metrics["n_mentions"]),
                "n_hallucinated": int(metrics["n_hallucinated"]),
            }
        )
    return rows
