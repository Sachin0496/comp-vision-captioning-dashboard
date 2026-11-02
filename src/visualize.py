"""Publication-style visualizations for a COCO image-captioning project.

The project compares three captioners:

``global``
    A CNN-LSTM captioner without attention.
``attention``
    A CNN-LSTM captioner with Bahdanau (additive) attention.
``object``
    The attention captioner augmented with detected object labels.

This module is deliberately self-contained: it depends only on ``numpy``,
``Pillow`` and ``matplotlib``.  All functions render headlessly (the caller is
responsible for selecting a non-interactive matplotlib backend, e.g. by
setting ``MPLBACKEND=Agg``) and close their figures after saving.

Public functions
----------------
overlay_attention
    Draw one attention map on top of an image.
visualize_attention_samples
    Grid of images, captions and per-word attention maps.
plot_model_comparison
    BLEU-4 and CHAIR bars for all models.
plot_bleu_hallucination
    Sentence-BLEU vs. number of hallucinated objects, with a binned trend.
plot_training_curves
    Loss and validation-metric curves.
plot_qualitative_examples
    Image / model-caption grid highlighting hallucinated words.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

__all__ = [
    "MODEL_COLORS",
    "MODEL_LABELS",
    "overlay_attention",
    "visualize_attention_samples",
    "plot_model_comparison",
    "plot_bleu_hallucination",
    "plot_training_curves",
    "plot_qualitative_examples",
]

# ---------------------------------------------------------------------------
# Palette and typography
# ---------------------------------------------------------------------------

MODEL_COLORS: dict[str, str] = {
    "global": "#4C72B0",
    "attention": "#DD8452",
    "object": "#55A868",
}

MODEL_LABELS: dict[str, str] = {
    "global": "Global (CNN-LSTM)",
    "attention": "Attention (Bahdanau)",
    "object": "Object-aware",
}

# Stable fallback colours for models that are not part of the core comparison.
_FALLBACK_COLORS = (
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
)

_TEXT_COLOR = "#222222"
_MUTED_COLOR = "#666666"
_GRID_COLOR = "#DDDDDD"
_HALLUCINATION_COLOR = "#C0392B"

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and",
        "or", "but", "nor", "is", "am", "are", "was", "were", "be", "been",
        "being", "has", "have", "had", "do", "does", "did", "doing", "this",
        "that", "these", "those", "there", "here", "it", "its", "they", "them",
        "their", "theirs", "he", "him", "his", "she", "her", "hers", "we",
        "us", "our", "ours", "you", "your", "yours", "i", "me", "my", "mine",
        "as", "by", "from", "into", "onto", "over", "under", "near", "next",
        "beside", "behind", "above", "below", "up", "down", "out", "off",
        "than", "then", "so", "such", "very", "just", "also", "not", "no",
        "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "some", "any", "each", "every", "both", "all", "more", "most",
        "few", "many", "much", "other", "another", "one", "two", "three",
        "while", "when", "where", "who", "whom", "whose", "which", "what",
        "how", "why", "if", "because", "about", "across", "along", "among",
        "around", "before", "after", "during", "through", "between", "against",
        "upon", "within", "without", "toward", "towards", "per", "via",
        "s", "t", "re", "ve", "ll", "d", "m",
    }
)

_SPECIAL_TOKEN_RE = re.compile(r"^(?:<[^>]*>|\[[^\]]*\]|\{[^}]*\})$")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")


# ---------------------------------------------------------------------------
# Small private helpers
# ---------------------------------------------------------------------------


def _model_color(model: str) -> str:
    """Return a stable colour for ``model`` (known models use the palette)."""

    key = str(model).strip().lower()
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]
    # Deterministic fallback: independent of Python's hash randomisation.
    if not key:
        return _FALLBACK_COLORS[0]
    return _FALLBACK_COLORS[sum(ord(ch) for ch in key) % len(_FALLBACK_COLORS)]


def _model_label(model: str) -> str:
    """Return a human-readable label for a model name."""

    key = str(model).strip().lower()
    if key in MODEL_LABELS:
        return MODEL_LABELS[key]
    return str(model).replace("_", " ").strip().title()


def _display_word(word: Any) -> str:
    """Clean a generated word for use as a plot title."""

    return str(word).strip().strip("\"'` ")


def _clean_token(word: Any) -> str:
    """Lower-case token stripped of surrounding punctuation."""

    token = str(word).strip().lower()
    token = token.strip("\"'`.,!?;:()[]{}<>-—–…")
    return token


def _is_content_word(word: Any) -> bool:
    """Heuristic noun/verb/adjective detector for caption words.

    No POS tagger is available in the project's minimal environment, so this
    uses a conservative stopword + punctuation filter.  The result is intended
    for selecting readable attention words, not for linguistic analysis.
    """

    raw = str(word).strip()
    if not raw or _SPECIAL_TOKEN_RE.match(raw):
        return False
    cleaned = raw.strip("\"'`.,!?;:()[]{}<>-—–…")
    if len(cleaned) < 2 or not re.search(r"[A-Za-z]", cleaned):
        return False
    if _clean_token(cleaned) in _STOPWORDS:
        return False
    return True


def _select_word_indices(words: Sequence[Any], max_words: int) -> list[int]:
    """Pick up to ``max_words`` content words spread across the caption."""

    if max_words is None or int(max_words) <= 0 or not words:
        return []

    candidates = [i for i, word in enumerate(words) if _is_content_word(word)]
    if not candidates:
        # Degenerate captions (all stopwords/punctuation): keep something.
        candidates = [i for i, word in enumerate(words) if _display_word(word)]

    limit = min(int(max_words), len(candidates))
    if limit <= 0:
        return []
    if len(candidates) <= limit:
        return candidates

    # Evenly spaced positions across the candidate list, then map back to the
    # original word indices so the selection spans the whole caption.
    positions = np.linspace(0, len(candidates) - 1, num=limit)
    selected: list[int] = []
    for position in positions:
        index = candidates[int(np.clip(round(float(position)), 0, len(candidates) - 1))]
        if index not in selected:
            selected.append(index)

    # Rounding can only rarely produce duplicates; fill the remainder in order.
    if len(selected) < limit:
        for index in candidates:
            if index not in selected:
                selected.append(index)
            if len(selected) == limit:
                break
    return sorted(selected)


def _normalize_attention(att_map: np.ndarray) -> np.ndarray:
    """Normalise a 2-D attention map to ``[0, 1]``; constant maps -> zeros."""

    array = np.asarray(att_map, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"att_map must be a 2-D array, got shape {array.shape}")
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, 0.0, None)
    low = float(np.min(array))
    high = float(np.max(array))
    if not np.isfinite(low) or not np.isfinite(high) or (high - low) < 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def _resize_attention(att_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Normalise an attention map and resize it to ``size=(width, height)``."""

    normalised = _normalize_attention(att_map)
    if normalised.shape == (size[1], size[0]):
        return normalised.astype(np.float32, copy=False)
    pil_map = Image.fromarray(normalised.astype(np.float32))
    resized = pil_map.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _load_rgb(image_path: Any) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array (always a writable copy)."""

    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def _draw_overlay(
    ax: Axes,
    image: np.ndarray,
    attention: np.ndarray,
    *,
    cmap: str,
    alpha: float,
    title: str | None,
) -> None:
    """Draw an image plus a resized attention map on an existing axis."""

    ax.imshow(image)
    ax.imshow(
        attention,
        cmap=cmap,
        alpha=float(np.clip(alpha, 0.0, 1.0)),
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax.set_axis_off()
    if title:
        ax.set_title(str(title), fontsize=10, fontweight="bold", pad=6)


def _style_axis(ax: Axes, *, grid_axis: str = "y") -> None:
    """Apply the shared clean publication style to an axis."""

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#444444", labelsize=9)
    ax.grid(
        axis=grid_axis,
        color=_GRID_COLOR,
        linewidth=0.7,
        alpha=0.9,
        zorder=0,
    )
    ax.set_axisbelow(True)


def _save_figure(fig: Figure, out_path: Any, dpi: int) -> str:
    """Save a figure (creating parents), close it and return the path string."""

    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _series(records: Sequence[Mapping[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted, finite ``(epoch, value)`` pairs for one metric key."""

    pairs: list[tuple[float, float]] = []
    for record in records:
        if record is None:
            continue
        epoch = record.get("epoch")
        value = record.get(key)
        if epoch is None or value is None:
            continue
        try:
            x = float(epoch)
            y = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    if not pairs:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    pairs.sort(key=lambda item: item[0])
    epochs, values = zip(*pairs)
    return np.asarray(epochs, dtype=float), np.asarray(values, dtype=float)


def _binned_trend(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-frequency binning with bootstrap 95% CIs for the bin means.

    Returns ``(bin_x, bin_mean, lower_error, upper_error)`` ready for
    :meth:`matplotlib.axes.Axes.errorbar`.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty.copy(), empty.copy(), empty.copy()

    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]

    bins = int(max(1, min(n_bins, x_sorted.size)))
    groups = np.array_split(np.arange(x_sorted.size), bins)
    rng = np.random.default_rng(seed)

    bin_x: list[float] = []
    bin_mean: list[float] = []
    lower_err: list[float] = []
    upper_err: list[float] = []
    for group in groups:
        if group.size == 0:
            continue
        values = y_sorted[group]
        mean = float(np.mean(values))
        if group.size == 1 or n_bootstrap <= 0:
            low = high = mean
        else:
            samples = rng.choice(
                values, size=(int(n_bootstrap), values.size), replace=True
            ).mean(axis=1)
            low, high = (float(v) for v in np.percentile(samples, [2.5, 97.5]))
        bin_x.append(float(np.mean(x_sorted[group])))
        bin_mean.append(mean)
        lower_err.append(max(0.0, mean - low))
        upper_err.append(max(0.0, high - mean))

    return (
        np.asarray(bin_x, dtype=float),
        np.asarray(bin_mean, dtype=float),
        np.asarray(lower_err, dtype=float),
        np.asarray(upper_err, dtype=float),
    )


def _hallucination_token_set(items: Sequence[Any] | None) -> set[str]:
    """Normalise hallucinated words/phrases into a set of content tokens."""

    tokens: set[str] = set()
    for item in items or []:
        for match in _WORD_RE.findall(str(item).lower()):
            if match not in _STOPWORDS:
                tokens.add(match)
    return tokens


def _draw_text_panel(
    ax: Axes,
    model: str,
    caption: str,
    hallucinated: Sequence[Any] | None,
) -> None:
    """Render a model caption panel, highlighting hallucinated words in red."""

    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    color = _model_color(model)
    header = TextArea(
        _model_label(model),
        textprops={"color": color, "fontsize": 10, "fontweight": "bold"},
    )

    caption = str(caption or "").strip() or "(no caption generated)"
    wrapped_lines = textwrap.wrap(caption, width=34) or [caption]
    hall_tokens = _hallucination_token_set(hallucinated)

    line_boxes = []
    for line in wrapped_lines:
        words = line.split(" ")
        word_boxes = []
        for position, word in enumerate(words):
            suffix = " " if position < len(words) - 1 else ""
            is_hallucinated = _clean_token(word) in hall_tokens
            word_boxes.append(
                TextArea(
                    word + suffix,
                    textprops={
                        "color": _HALLUCINATION_COLOR if is_hallucinated else _TEXT_COLOR,
                        "fontsize": 9.2,
                        "fontweight": "bold" if is_hallucinated else "normal",
                    },
                )
            )
        line_boxes.append(
            HPacker(children=word_boxes, align="baseline", pad=0, sep=0)
        )

    caption_box = VPacker(children=line_boxes, align="left", pad=0, sep=1.5)

    children: list[Any] = [header, caption_box]
    hall_terms = [str(item).strip() for item in (hallucinated or []) if str(item).strip()]
    if hall_terms:
        children.append(
            TextArea(
                "Hallucinated: " + ", ".join(hall_terms),
                textprops={
                    "color": _HALLUCINATION_COLOR,
                    "fontsize": 8.0,
                    "style": "italic",
                },
            )
        )

    content = VPacker(children=children, align="left", pad=0, sep=5)
    box = AnnotationBbox(
        content,
        (0.03, 0.97),
        xycoords="axes fraction",
        box_alignment=(0.0, 1.0),
        frameon=True,
        bboxprops={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#F8F8F8",
            "edgecolor": "#E3E3E3",
            "linewidth": 0.7,
        },
        zorder=5,
    )
    ax.add_artist(box)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def overlay_attention(
    image_path: Any,
    att_map: np.ndarray,
    ax: Axes | None = None,
    cmap: str = "inferno",
    alpha: float = 0.55,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Overlay a normalised attention map on an image.

    Parameters
    ----------
    image_path:
        Path to an RGB image (any format understood by Pillow).
    att_map:
        2-D array of non-negative attention weights (typically ``7 x 7``).
        It is robustly min-max normalised (a constant map becomes all zeros)
        and resized to the image size with Pillow's bilinear filter.
    ax:
        Optional existing axis.  If omitted a new figure/axis is created.
    cmap:
        Matplotlib colormap for the attention heatmap.
    alpha:
        Opacity of the heatmap in ``[0, 1]``.
    title:
        Optional axis title.

    Returns
    -------
    (figure, axis)
        The figure containing the drawing and the axis used.
    """

    image = _load_rgb(image_path)
    height, width = image.shape[:2]
    attention = _resize_attention(att_map, (width, height))

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    else:
        fig = ax.figure

    _draw_overlay(ax, image, attention, cmap=cmap, alpha=alpha, title=title)
    return fig, ax


def visualize_attention_samples(
    samples: Sequence[Mapping[str, Any]],
    out_path: Any,
    max_words: int = 5,
    dpi: int = 170,
) -> str:
    """Plot a grid of captions with the attention map for selected words.

    Each row corresponds to one sample.  The first column shows the original
    image with the complete caption wrapped underneath it, followed by one
    column per selected word.

    Parameters
    ----------
    samples:
        Dictionaries with keys ``image_path``, ``caption``, ``words`` and
        ``attention`` (``[T, 7, 7]``, one map per generated word).  An optional
        ``highlight`` key overrides the automatic word selection with explicit
        word indices.
    out_path:
        Output image path.  Parent directories are created.
    max_words:
        Maximum number of words selected automatically per sample.
    dpi:
        Output resolution.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If ``samples`` is empty or an attention array has an invalid shape.
    """

    samples = list(samples)
    if not samples:
        raise ValueError("samples must contain at least one sample")

    prepared: list[dict[str, Any]] = []
    max_selected = 0

    for sample in samples:
        image_path = sample["image_path"]
        caption = str(sample.get("caption", "") or "")
        words = [_display_word(word) for word in (sample.get("words") or [])]

        attention_value = sample.get("attention")
        attention: np.ndarray | None
        if attention_value is None:
            attention = None
            n_maps = len(words)
        else:
            attention = np.asarray(attention_value)
            if attention.ndim == 2:
                attention = attention[None, ...]
            if attention.ndim != 3:
                raise ValueError(
                    "attention must have shape [T, H, W]; "
                    f"got {attention.shape} for {image_path!r}"
                )
            n_maps = int(attention.shape[0])

        n_words = min(len(words), n_maps) if words else n_maps

        highlight = sample.get("highlight")
        if highlight is None:
            selected = [i for i in _select_word_indices(words, max_words) if i < n_words]
        else:
            selected = []
            for raw_index in highlight:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < n_words and index not in selected:
                    selected.append(index)

        prepared.append(
            {
                "image_path": image_path,
                "caption": caption,
                "words": words,
                "attention": attention,
                "selected": selected,
            }
        )
        max_selected = max(max_selected, len(selected))

    n_rows = len(prepared)
    n_cols = 1 + max_selected
    fig_width = max(4.2, 2.9 + 2.25 * max_selected)
    fig_height = max(2.8, 2.8 * n_rows)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    try:
        outer = fig.add_gridspec(
            n_rows,
            1,
            hspace=0.20,
            top=0.975,
            bottom=0.025,
            left=0.015,
            right=0.985,
        )

        for row, sample in enumerate(prepared):
            # All image/word panels share one row of equal-width axes so their
            # (aspect-constrained) images line up perfectly.  The caption gets
            # a dedicated strip directly underneath the first column.
            row_grid = outer[row, 0].subgridspec(
                2,
                n_cols,
                height_ratios=[4.4, 1.0],
                hspace=0.04,
                wspace=0.10,
            )
            ax_image = fig.add_subplot(row_grid[0, 0])
            ax_caption = fig.add_subplot(row_grid[1, 0])

            image = _load_rgb(sample["image_path"])
            ax_image.imshow(image)
            ax_image.set_axis_off()

            caption = sample["caption"] or "(no caption)"
            wrapped_caption = textwrap.fill(caption, width=36)
            ax_caption.set_axis_off()
            ax_caption.text(
                0.0,
                0.98,
                wrapped_caption,
                transform=ax_caption.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color=_TEXT_COLOR,
                linespacing=1.18,
            )

            words = sample["words"]
            attention = sample["attention"]
            selected = sample["selected"]
            for column, word_index in enumerate(selected):
                ax_word = fig.add_subplot(row_grid[0, column + 1])
                title = f'"{words[word_index]}"'
                if attention is not None and word_index < attention.shape[0]:
                    attention_map = attention[word_index]
                    weights = _resize_attention(
                        attention_map, (image.shape[1], image.shape[0])
                    )
                    _draw_overlay(
                        ax_word,
                        image,
                        weights,
                        cmap="inferno",
                        alpha=0.55,
                        title=title,
                    )
                else:
                    ax_word.imshow(image)
                    ax_word.set_axis_off()
                    ax_word.set_title(
                        title, fontsize=10, fontweight="bold", pad=6
                    )

        return _save_figure(fig, out_path, dpi)
    except Exception:
        plt.close(fig)
        raise


def plot_model_comparison(
    summary: Sequence[Mapping[str, Any]],
    out_path: Any,
    dpi: int = 170,
) -> str:
    """Compare models on BLEU-4 and CHAIR_s / CHAIR_i with horizontal bars.

    Parameters
    ----------
    summary:
        One dictionary per model with keys ``model``, ``bleu4``, ``chair_s``
        and optionally ``chair_i``.  BLEU and CHAIR values are percentages
        (CHAIR in ``[0, 100]``).
    out_path:
        Output image path.  Parent directories are created.
    dpi:
        Output resolution.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If ``summary`` is empty.
    """

    summary = list(summary)
    if not summary:
        raise ValueError("summary must contain at least one model")

    models = [str(item.get("model", f"model {i}")) for i, item in enumerate(summary)]
    n_models = len(models)
    positions = np.arange(n_models)
    colors = [_model_color(model) for model in models]

    try:
        bleu4 = np.asarray([float(item.get("bleu4", 0.0)) for item in summary], dtype=float)
        chair_s = np.asarray([float(item.get("chair_s", 0.0)) for item in summary], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bleu4 and chair_s must be numeric: {exc}") from exc

    has_chair_i = any(
        item.get("chair_i") is not None for item in summary
    )
    chair_i = np.asarray(
        [
            float(item["chair_i"]) if item.get("chair_i") is not None else np.nan
            for item in summary
        ],
        dtype=float,
    )

    fig_height = max(2.8, 0.66 * n_models + 1.55)
    fig, (ax_bleu, ax_chair) = plt.subplots(
        1, 2, figsize=(8.0, fig_height), sharey=True
    )

    try:
        window = 0.64
        ax_bleu.barh(
            positions,
            bleu4,
            height=window,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        bleu_max = max(1.0, float(np.nanmax(bleu4)) if bleu4.size else 1.0)
        ax_bleu.set_xlim(0.0, bleu_max * 1.24)
        for y, value in zip(positions, bleu4):
            ax_bleu.text(
                value + bleu_max * 0.02,
                y,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=8.2,
                color="#333333",
            )
        ax_bleu.set_yticks(positions)
        ax_bleu.set_yticklabels([_model_label(model) for model in models], fontsize=8.4)
        ax_bleu.set_xlabel("BLEU-4 (%, higher is better)", fontsize=8.8)
        ax_bleu.set_title("Caption quality", fontsize=10.4, fontweight="bold", pad=7)
        _style_axis(ax_bleu, grid_axis="x")

        if has_chair_i:
            height = 0.34
            y_sentence = positions - 0.18
            y_instance = positions + 0.18
            ax_chair.barh(
                y_sentence,
                chair_s,
                height=height,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax_chair.barh(
                y_instance,
                chair_i,
                height=height,
                color=colors,
                edgecolor="#555555",
                linewidth=0.5,
                alpha=0.45,
                hatch="///",
                zorder=3,
            )
            legend_handles = [
                Patch(facecolor="#888888", label=r"CHAIR$_s$ (sentence)"),
                Patch(
                    facecolor="#B0B0B0",
                    edgecolor="#555555",
                    hatch="///",
                    label=r"CHAIR$_i$ (instance)",
                ),
            ]
            ax_chair.legend(
                handles=legend_handles,
                loc="lower right",
                frameon=False,
                fontsize=7.3,
            )
        else:
            ax_chair.barh(
                positions,
                chair_s,
                height=0.64,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

        chair_max = max(1.0, float(np.nanmax(chair_s)) if chair_s.size else 1.0)
        if has_chair_i and np.isfinite(chair_i).any():
            chair_max = max(chair_max, float(np.nanmax(chair_i)))
        ax_chair.set_xlim(0.0, chair_max * 1.26)

        if has_chair_i:
            for y, value in zip(positions - 0.18, chair_s):
                ax_chair.text(
                    value + chair_max * 0.015,
                    y,
                    f"{value:.1f}",
                    va="center",
                    ha="left",
                    fontsize=7.8,
                    color="#333333",
                )
            for y, value in zip(positions + 0.18, chair_i):
                if np.isfinite(value):
                    ax_chair.text(
                        value + chair_max * 0.015,
                        y,
                        f"{value:.1f}",
                        va="center",
                        ha="left",
                        fontsize=7.4,
                        color="#555555",
                    )
        else:
            for y, value in zip(positions, chair_s):
                ax_chair.text(
                    value + chair_max * 0.02,
                    y,
                    f"{value:.1f}",
                    va="center",
                    ha="left",
                    fontsize=8.2,
                    color="#333333",
                )

        ax_chair.set_xlabel(
            r"CHAIR hallucination rate (%, lower is better)", fontsize=8.8
        )
        ax_chair.set_title(
            "Object hallucination", fontsize=10.4, fontweight="bold", pad=7
        )
        _style_axis(ax_chair, grid_axis="x")
        ax_chair.tick_params(axis="y", left=False, labelleft=False)

        ax_bleu.invert_yaxis()
        fig.suptitle(
            "Model comparison", fontsize=11.8, fontweight="bold", y=0.99
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        return _save_figure(fig, out_path, dpi)
    except Exception:
        plt.close(fig)
        raise


def plot_bleu_hallucination(
    records: Sequence[Mapping[str, Any]],
    out_path: Any,
    dpi: int = 170,
) -> str:
    """Scatter sentence-BLEU against hallucinated objects, with a trend.

    A binned trend line uses five equal-frequency BLEU bins.  Each bin shows
    the mean number of hallucinated objects with a 95% bootstrap confidence
    interval computed over all records.

    Parameters
    ----------
    records:
        Dictionaries with keys ``model``, ``sentence_bleu4`` (0-100),
        ``n_hallucinated`` (int) and optionally ``has_hallucination`` (bool).
    out_path:
        Output image path.  Parent directories are created.
    dpi:
        Output resolution.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If ``records`` is empty or contains no finite numeric points.
    """

    records = list(records)
    if not records:
        raise ValueError("records must be non-empty")

    models: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    point_models: list[str] = []
    for record in records:
        try:
            x = float(record["sentence_bleu4"])
            y = float(record["n_hallucinated"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        model = str(record.get("model", "unknown"))
        if model not in models:
            models.append(model)
        xs.append(x)
        ys.append(y)
        point_models.append(model)

    if not xs:
        raise ValueError("records contain no finite sentence_bleu4/n_hallucinated points")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    model_array = np.asarray(point_models, dtype=object)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    try:
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.18, 0.18, size=x.size)

        for model in models:
            mask = model_array == model
            ax.scatter(
                x[mask],
                y[mask] + jitter[mask],
                s=46,
                color=_model_color(model),
                alpha=0.72,
                edgecolors="white",
                linewidths=0.6,
                label=_model_label(model),
                zorder=3,
            )

        bin_x, bin_mean, lower_error, upper_error = _binned_trend(x, y)
        ax.errorbar(
            bin_x,
            bin_mean,
            yerr=np.vstack([lower_error, upper_error]),
            fmt="o-",
            color="#2B2B2B",
            ecolor="#2B2B2B",
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.2,
            markersize=5.5,
            linewidth=1.8,
            label="Binned mean (95% bootstrap CI)",
            zorder=4,
        )

        ax.set_xlabel("Sentence BLEU-4 (%)", fontsize=10.5)
        ax.set_ylabel("Hallucinated objects per caption", fontsize=10.5)
        ax.set_title(
            "Caption quality vs. object hallucination",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        x_low = max(0.0, float(np.floor(x.min())) - 4.0)
        x_high = min(100.0, float(np.ceil(x.max())) + 4.0)
        if x_high - x_low < 10.0:
            midpoint = 0.5 * (x_low + x_high)
            x_low = max(0.0, midpoint - 5.0)
            x_high = min(100.0, midpoint + 5.0)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(-0.6, max(1.2, float(y.max()) + 1.0))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axis(ax, grid_axis="both")
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=8.8,
            scatterpoints=1,
            handletextpad=0.2,
        )
        fig.tight_layout()
        return _save_figure(fig, out_path, dpi)
    except Exception:
        plt.close(fig)
        raise


def plot_training_curves(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    out_path: Any,
    dpi: int = 170,
) -> str:
    """Plot train/validation loss and, when present, validation metrics.

    Parameters
    ----------
    histories:
        Mapping ``model name -> list of epoch records``.  Every record must
        contain ``epoch``, ``train_loss`` and ``val_loss``; ``val_bleu4`` and
        ``val_chair_s`` are optional.
    out_path:
        Output image path.  Parent directories are created.
    dpi:
        Output resolution.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If no usable history records are supplied.
    """

    if not histories:
        raise ValueError("histories must contain at least one model")

    prepared: dict[str, list[Mapping[str, Any]]] = {}
    for model, records in histories.items():
        usable = [record for record in (records or []) if record is not None]
        if usable:
            prepared[str(model)] = usable
    if not prepared:
        raise ValueError("histories contain no records")

    has_bleu = any(
        any(record.get("val_bleu4") is not None for record in records)
        for records in prepared.values()
    )
    has_chair = any(
        any(record.get("val_chair_s") is not None for record in records)
        for records in prepared.values()
    )
    has_metrics = has_bleu or has_chair
    metric_specs: list[tuple[str, str, str]] = []
    if has_bleu:
        metric_specs.append(("val_bleu4", "Validation BLEU-4", "BLEU-4 (%)"))
    if has_chair:
        metric_specs.append(("val_chair_s", "Validation CHAIR$_s$", "CHAIR$_s$ (%)"))

    if has_metrics and len(metric_specs) == 1:
        fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.35))
        ax_train, ax_val = axes[0], axes[1]
        bottom_axes = [axes[2]]
    elif has_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.9))
        ax_train, ax_val = axes[0, 0], axes[0, 1]
        bottom_axes = [axes[1, 0], axes[1, 1]]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
        ax_train, ax_val = axes[0], axes[1]
        bottom_axes = []

    try:
        for model, records in prepared.items():
            color = _model_color(model)
            label = _model_label(model)
            epochs, train_loss = _series(records, "train_loss")
            if epochs.size:
                ax_train.plot(
                    epochs,
                    train_loss,
                    color=color,
                    linewidth=1.9,
                    marker="o",
                    markersize=3.4,
                    label=label,
                )
            val_epochs, val_loss = _series(records, "val_loss")
            if val_epochs.size:
                ax_val.plot(
                    val_epochs,
                    val_loss,
                    color=color,
                    linewidth=1.9,
                    linestyle="--",
                    marker="o",
                    markersize=3.4,
                    label=label,
                )

        ax_train.set_title("Training loss", fontsize=9.8, fontweight="bold")
        ax_val.set_title("Validation loss", fontsize=9.8, fontweight="bold")
        for axis in (ax_train, ax_val):
            axis.set_xlabel("Epoch", fontsize=8.5)
            axis.set_ylabel("Cross-entropy loss", fontsize=8.5)
            axis.set_ylim(bottom=0.0)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            _style_axis(axis, grid_axis="both")
            axis.legend(frameon=False, fontsize=7.4)

        for axis, (key, title, ylabel) in zip(bottom_axes, metric_specs):
            plotted = False
            for model, records in prepared.items():
                epochs, values = _series(records, key)
                if not epochs.size:
                    continue
                plotted = True
                axis.plot(
                    epochs,
                    values,
                    color=_model_color(model),
                    linewidth=1.9,
                    marker="o",
                    markersize=3.6,
                    label=_model_label(model),
                )
            axis.set_title(title, fontsize=9.8, fontweight="bold")
            axis.set_xlabel("Epoch", fontsize=8.5)
            axis.set_ylabel(ylabel, fontsize=8.5)
            axis.set_ylim(bottom=0.0)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            _style_axis(axis, grid_axis="both")
            if plotted:
                axis.legend(frameon=False, fontsize=7.4)

        # Hide unused bottom panels when only one metric (or none) exists.
        for axis in bottom_axes[len(metric_specs):]:
            axis.set_axis_off()

        fig.suptitle(
            "Training dynamics", fontsize=11.5, fontweight="bold", y=0.995
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, has_metrics * 0.04 + 0.95))
        return _save_figure(fig, out_path, dpi)
    except Exception:
        plt.close(fig)
        raise


def plot_qualitative_examples(
    examples: Sequence[Mapping[str, Any]],
    out_path: Any,
    dpi: int = 170,
) -> str:
    """Show images, ground-truth objects and each model's caption side by side.

    Hallucinated words are rendered in red and a compact red note lists the
    hallucinated objects when the optional ``hallucinated`` mapping is given.

    Parameters
    ----------
    examples:
        Dictionaries with keys ``image_path``, ``references`` (list of str),
        ``captions`` (mapping model -> str), ``gt_objects`` (list of str) and
        optionally ``hallucinated`` (mapping model -> list of str).
    out_path:
        Output image path.  Parent directories are created.
    dpi:
        Output resolution.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If ``examples`` is empty.
    """

    examples = list(examples)
    if not examples:
        raise ValueError("examples must contain at least one image")

    models: list[str] = []
    for example in examples:
        captions = example.get("captions") or {}
        for model in captions:
            if str(model) not in models:
                models.append(str(model))

    n_models = len(models)
    n_rows = len(examples)
    n_cols = 1 + n_models
    fig_width = 3.15 + 2.55 * n_models
    fig_height = max(2.8, 2.8 * n_rows)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    try:
        outer = fig.add_gridspec(
            n_rows,
            n_cols,
            width_ratios=[1.2] + [1.0] * n_models,
            hspace=0.16,
            wspace=0.16,
            top=0.975,
            bottom=0.02,
            left=0.015,
            right=0.985,
        )

        for row, example in enumerate(examples):
            image_grid = outer[row, 0].subgridspec(
                2, 1, height_ratios=[4.8, 1.1], hspace=0.0
            )
            ax_image = fig.add_subplot(image_grid[0])
            ax_objects = fig.add_subplot(image_grid[1])

            image = _load_rgb(example["image_path"])
            ax_image.imshow(image)
            ax_image.set_axis_off()

            objects = [str(item) for item in (example.get("gt_objects") or [])]
            objects_text = (
                "GT objects: " + ", ".join(objects) if objects else "GT objects: (none)"
            )
            ax_objects.set_axis_off()
            ax_objects.text(
                0.0,
                0.98,
                textwrap.fill(objects_text, width=38),
                transform=ax_objects.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color=_MUTED_COLOR,
                linespacing=1.12,
            )

            captions = example.get("captions") or {}
            hallucinated_map = example.get("hallucinated") or {}
            for column, model in enumerate(models, start=1):
                ax_text = fig.add_subplot(outer[row, column])
                _draw_text_panel(
                    ax_text,
                    model,
                    str(captions.get(model, "(no caption)") or ""),
                    hallucinated_map.get(model),
                )

        return _save_figure(fig, out_path, dpi)
    except Exception:
        plt.close(fig)
        raise
