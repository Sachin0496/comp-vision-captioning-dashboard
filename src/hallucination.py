"""Object-hallucination scoring for COCO image captioning (CHAIR-style).

This module implements the evaluation core used to answer the project's research
question: *do generated captions mention objects that are not actually present in
the image?*  It extracts COCO-object mentions from free-form captions, compares
them against per-image ground-truth object lists, and reports CHAIR-style rates.

Metrics
-------
CHAIR-S (sentence / aggregate rate)
    ``sum(hallucinated object mentions) / sum(all object mentions)`` over the
    whole test set.  Each caption contributes the *unique set* of COCO classes
    it names, so repeating "dog ... dog" in one caption counts once.  This is
    the standard CHAIR_S aggregation of Rohrbach et al. (2018).

CHAIR-I (image / per-image rate)
    Fraction of images whose caption contains at least one hallucinated object
    mention.  This is the standard CHAIR_I definition.

We additionally report micro object precision / recall / F1 over the same
mention sets, which makes the precision/recall trade-off explicit: a model that
says nothing gets recall 0, while a model that lists every class gets
precision 0.

Matching
--------
Mentions are found with a lowercased, word-boundary, longest-phrase-first
regular expression over the curated :data:`SYNONYMS` table, without overlapping
matches.  That is what makes "hot dog" count as ``hot dog`` *and not* ``dog``,
and "teddy bear" count as ``teddy bear`` *and not* ``bear``.

Key caveat
----------
COCO instance annotations are **incomplete**: small, occluded, or background
objects are frequently unlabelled.  A mention missing from the COCO ground
truth is therefore only *evidence* of hallucination, not proof -- the measured
rate is an upper bound on true hallucination.  Only the 80 COCO object
categories are scored; other nouns (sky, grass, water, street, tree, ...) are
deliberately ignored, both because they carry no annotation and because
captions legitimately describe scene context COCO does not label.

References
----------
Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., & Saenko, K. (2018).
*Object Hallucination in Image Captioning.* EMNLP 2018.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "COCO_CLASSES",
    "COCO_CLASS_SET",
    "SYNONYMS",
    "extract_mentions",
    "mention_summary",
    "score_hallucination",
]

# --------------------------------------------------------------------------- #
# COCO categories
# --------------------------------------------------------------------------- #

#: The canonical 80 COCO object categories in the standard COCO ordering (the
#: order of ``categories`` in ``instances_val2017.json``).
COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

#: Fast membership test for canonical class names.
COCO_CLASS_SET: frozenset[str] = frozenset(COCO_CLASSES)


# --------------------------------------------------------------------------- #
# Surface form -> canonical COCO class
# --------------------------------------------------------------------------- #
# Curated by hand.  Guiding rule: map a surface form only when, *in COCO
# captioning text*, it overwhelmingly denotes the object rather than a scene
# attribute, an action, or a different sense of the word.
#
# Deliberately NOT mapped (ambiguous or not COCO objects): sky, grass, water,
# street, road, tree, building, wall, floor, ground, field, ocean, beach, snow,
# mountain, room, kitchen, crowd, screen, monitor, bag, trunk, seat, sign,
# board, mobile, "computer" (desktop/device sense is too ambiguous).
#
# Judgement calls, documented:
#   * "table"     -> dining table   (only table class in COCO)
#   * "glass"     -> wine glass     ("sunglasses" is excluded by the regex)
#   * "bat"       -> baseball bat   (accepted ambiguity)
#   * "plant"     -> potted plant   (only plant class in COCO)
#   * agent nouns ("rider", "player", "cyclist", ...) -> person
_SYNONYM_GROUPS: dict[str, tuple[str, ...]] = {
    # -- person -----------------------------------------------------------
    "person": (
        "person",
        "people",
        "man",
        "men",
        "woman",
        "women",
        "boy",
        "girl",
        "child",
        "children",
        "kid",
        "guy",
        "lady",
        "ladies",
        "gentleman",
        "gentlemen",
        "human",
        "pedestrian",
        "baby",
        "toddler",
        "rider",
        "player",
        "driver",
        "skier",
        "surfer",
        "skateboarder",
        "snowboarder",
        "cyclist",
        "biker",
        "jogger",
        "crowd",
    ),
    # -- vehicles ---------------------------------------------------------
    "bicycle": ("bicycle", "bike"),
    "car": ("car", "automobile", "sedan", "vehicle"),
    "motorcycle": ("motorcycle", "motorbike"),
    "airplane": ("airplane", "aeroplane", "plane", "jet", "aircraft", "jetliner"),
    "bus": ("bus", "school bus"),
    "train": ("train", "locomotive"),
    "truck": ("truck", "lorry", "pickup", "pickup truck", "fire truck", "dump truck"),
    "boat": (
        "boat",
        "ship",
        "sailboat",
        "canoe",
        "kayak",
        "rowboat",
        "motorboat",
        "yacht",
        "ferry",
    ),
    "traffic light": ("traffic light", "traffic signal", "stoplight"),
    "fire hydrant": ("fire hydrant", "hydrant", "fire plug"),
    "stop sign": ("stop sign",),
    "parking meter": ("parking meter",),
    # -- street furniture -------------------------------------------------
    "bench": ("bench", "park bench"),
    "chair": ("chair", "armchair", "folding chair", "dining chair"),
    "couch": ("couch", "sofa", "loveseat"),
    "dining table": ("dining table", "dinner table", "kitchen table", "table"),
    "potted plant": ("potted plant", "pot plant", "houseplant", "plant"),
    "bed": ("bed", "bunk bed"),
    "toilet": ("toilet", "toilet bowl", "commode"),
    "sink": ("sink", "washbasin", "wash basin"),
    # -- animals ----------------------------------------------------------
    "bird": ("bird",),
    "cat": ("cat", "kitten", "kitty"),
    "dog": ("dog", "puppy", "pup", "canine"),
    "horse": ("horse", "pony", "stallion", "mare"),
    "sheep": ("sheep", "lamb", "ewe"),
    "cow": ("cow", "cattle", "bull", "ox"),
    "elephant": ("elephant",),
    "bear": ("bear",),
    "zebra": ("zebra",),
    "giraffe": ("giraffe",),
    # -- accessories / sports --------------------------------------------
    "backpack": ("backpack", "back pack", "knapsack", "rucksack"),
    "umbrella": ("umbrella", "parasol"),
    "handbag": ("handbag", "purse", "pocketbook"),
    "tie": ("tie", "necktie", "neck tie", "bow tie"),
    "suitcase": ("suitcase", "suit case", "luggage"),
    "frisbee": ("frisbee", "flying disc"),
    "skis": ("skis", "ski", "pair of skis"),
    "snowboard": ("snowboard", "snow board"),
    "skateboard": ("skateboard", "skate board"),
    "surfboard": ("surfboard", "surf board"),
    "sports ball": (
        "sports ball",
        "ball",
        "soccer ball",
        "football",
        "basketball",
        "volleyball",
        "tennis ball",
    ),
    "kite": ("kite",),
    "baseball bat": ("baseball bat", "bat"),
    "baseball glove": ("baseball glove", "fielding glove", "glove", "mitt"),
    "tennis racket": ("tennis racket", "tennis racquet", "racket", "racquet"),
    # -- tableware --------------------------------------------------------
    "bottle": ("bottle", "water bottle", "beer bottle", "soda bottle"),
    "wine glass": ("wine glass", "wineglass", "goblet", "glass"),
    "cup": ("cup", "mug", "coffee cup", "teacup"),
    "fork": ("fork",),
    "knife": ("knife", "knives", "kitchen knife"),
    "spoon": ("spoon",),
    "bowl": ("bowl",),
    # -- food -------------------------------------------------------------
    "banana": ("banana",),
    "apple": ("apple",),
    "sandwich": ("sandwich",),
    "orange": ("orange",),
    "broccoli": ("broccoli",),
    "carrot": ("carrot",),
    "hot dog": ("hot dog", "hotdog"),
    "pizza": ("pizza", "pizza slice", "slice of pizza"),
    "donut": ("donut", "doughnut"),
    "cake": ("cake", "birthday cake"),
    # -- electronics / appliances ----------------------------------------
    "tv": ("tv", "television", "telly", "tv set", "television set", "flat screen tv"),
    "laptop": ("laptop", "laptop computer", "notebook computer"),
    "mouse": ("mouse", "mice", "computer mouse"),
    "remote": ("remote", "remote control", "clicker"),
    "keyboard": ("keyboard", "computer keyboard"),
    "cell phone": (
        "cell phone",
        "cellphone",
        "mobile phone",
        "cellular phone",
        "smartphone",
        "phone",
    ),
    "microwave": ("microwave", "microwave oven"),
    "oven": ("oven",),
    "toaster": ("toaster",),
    "refrigerator": ("refrigerator", "fridge"),
    # -- indoor objects ---------------------------------------------------
    "book": ("book",),
    "clock": ("clock", "wall clock", "alarm clock"),
    "vase": ("vase", "flower vase"),
    "scissors": ("scissors", "scissor", "pair of scissors"),
    "teddy bear": ("teddy bear", "teddy", "stuffed bear", "stuffed animal", "plush bear"),
    "hair drier": ("hair drier", "hair dryer", "hairdryer", "blow dryer", "blow drier"),
    "toothbrush": ("toothbrush",),
}


def _pluralize(word: str) -> str:
    """Return the regular English plural of a single lowercase word.

    Irregular forms are listed explicitly in ``_SYNONYM_GROUPS`` and therefore
    never depend on this helper.
    """
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if len(word) >= 2 and word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


#: Tokens that already are plural (or are mass nouns) and must not be
#: "pluralized" again -- e.g. "skis" -> "skises" would be wrong.
_NO_FURTHER_PLURAL: frozenset[str] = frozenset(
    {
        "cattle",
        "children",
        "gentlemen",
        "knives",
        "ladies",
        "men",
        "mice",
        "people",
        "scissors",
        "sheep",
        "skis",
        "women",
    }
)

#: Multiword phrases whose final token defeats the generic rule.
_PHRASE_PLURAL_OVERRIDES: dict[str, str] = {
    "wine glass": "wine glasses",
    "school bus": "school buses",
}


def _pluralize_phrase(phrase: str) -> str | None:
    """Pluralize the final word of a phrase; ``None`` when unsafe.

    Returns ``None`` for phrases whose last token is already plural, is a mass
    noun, or ends in "s" but is not handled by an explicit override, so that we
    never invent forms like "skises" or "glasses" (eyewear sense).
    """
    if phrase in _PHRASE_PLURAL_OVERRIDES:
        return _PHRASE_PLURAL_OVERRIDES[phrase]
    words = phrase.split()
    if not words:
        return None
    last = words[-1]
    if last in _NO_FURTHER_PLURAL or last.endswith("s"):
        return None
    words[-1] = _pluralize(last)
    return " ".join(words)


def _build_synonyms(groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Flatten ``groups`` into a surface -> canonical class map.

    Regular plurals are generated for both single words and phrases ("dog" ->
    "dogs", "hot dog" -> "hot dogs") so plurals need no hand-maintained list.
    Explicit entries always win, because generated forms are only added when
    absent; irregular plurals ("mice", "knives") are therefore preserved.
    """
    table: dict[str, str] = {}
    for canonical, surfaces in groups.items():
        for surface in surfaces:
            table[surface] = canonical
    for surface, canonical in list(table.items()):
        plural = _pluralize_phrase(surface)
        if plural is not None and plural != surface:
            table.setdefault(plural, canonical)
    return table


#: Surface form (lowercase, whitespace normalized) -> canonical COCO class.
SYNONYMS: dict[str, str] = _build_synonyms(_SYNONYM_GROUPS)

# Hyphenated variants of multiword phrases ("hot-dog", "tennis-racket").
_SURFACE_INDEX: dict[str, str] = dict(SYNONYMS)
for _surface, _canonical in SYNONYMS.items():
    if " " in _surface:
        _SURFACE_INDEX.setdefault(_surface.replace(" ", "-"), _canonical)


def _build_pattern(index: Mapping[str, str]) -> re.Pattern[str]:
    """Compile the longest-phrase-first, word-boundary mention regex.

    Alternatives are ordered by descending word count, then descending length.
    With Python's leftmost-first alternation this guarantees that at any start
    position the longest phrase wins, which is what prevents ``hot dog`` from
    being read as ``dog`` and ``teddy bear`` from being read as ``bear``.
    ASCII-only case folding keeps character offsets stable.
    """
    keys = sorted(index, key=lambda key: (-len(key.split()), -len(key), key))
    alternation = "|".join(re.escape(key) for key in keys)
    return re.compile(
        rf"(?<![0-9A-Za-z])({alternation})(?![0-9A-Za-z])",
        re.IGNORECASE | re.ASCII,
    )


_PATTERN: re.Pattern[str] = _build_pattern(_SURFACE_INDEX)

_WORD_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> str:
    """Lowercase, normalize dashes/apostrophes, and collapse whitespace."""
    text = text.lower().replace("\u2019", "'")
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_mentions(caption: str) -> list[dict[str, Any]]:
    """Extract COCO object mentions from a caption.

    Matching is case-insensitive, uses word boundaries, prefers the longest
    phrase at each position, and never returns overlapping matches.  Every
    occurrence is returned as its own record (use the ``class_name`` field to
    deduplicate when computing CHAIR).

    Args:
        caption: Free-form caption text.

    Returns:
        A list of dicts ordered by ``start``, each with ``surface`` (the text as
        written), ``class_name`` (the canonical COCO class) and half-open
        character offsets ``start``/``end`` into the *original* caption.
        Non-COCO nouns and scene words produce no mentions.

    Examples:
        >>> [m["class_name"] for m in extract_mentions("a hot dog on a dining table")]
        ['hot dog', 'dining table']
        >>> [m["class_name"] for m in extract_mentions("two men riding bicycles")]
        ['person', 'person', 'bicycle']
    """
    if not caption or not isinstance(caption, str):
        return []

    mentions: list[dict[str, Any]] = []
    for match in _PATTERN.finditer(caption):
        surface = match.group(0)
        key = _normalize_text(surface)
        canonical = _SURFACE_INDEX.get(key)
        if canonical is None:  # pragma: no cover - defensive
            continue
        mentions.append(
            {
                "surface": surface,
                "class_name": canonical,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return mentions


def mention_summary(captions: Iterable[str]) -> dict[str, Any]:
    """Diagnostic mention counts over a caption collection."""
    counts: Counter[str] = Counter()
    caption_list = list(captions)
    total = 0
    for caption in caption_list:
        mentions = extract_mentions(caption)
        total += len(mentions)
        counts.update(mention["class_name"] for mention in mentions)
    return {
        "captions": len(caption_list),
        "mentions": total,
        "by_class": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _lookup_gt(gt_objects: Mapping[Any, Sequence[str]], image_id: Any) -> list[str]:
    """Look up the ground-truth object list, tolerating str/int key mixes."""
    candidates: list[Any] = [str(image_id)]
    if isinstance(image_id, str):
        try:
            candidates.append(int(image_id))
        except ValueError:
            pass
    elif isinstance(image_id, int) and not isinstance(image_id, bool):
        candidates.append(image_id)

    for key in candidates:
        if key in gt_objects:
            value = gt_objects[key]
            if isinstance(value, str):
                return [value]
            return [] if value is None else list(value)
    return []


def _normalize_gt(objects: Iterable[str]) -> list[str]:
    """Keep only canonical COCO classes, lowercased, deduplicated and sorted."""
    found: set[str] = set()
    for name in objects:
        if not isinstance(name, str):
            continue
        key = re.sub(r"\s+", " ", name.strip().lower())
        if key in COCO_CLASS_SET:
            found.add(key)
    return sorted(found)


def _to_native(value: Any) -> Any:
    """Convert numpy scalars to native Python types (JSON safety)."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            return value
    return value


def score_hallucination(
    predictions: Sequence[Mapping[str, Any]],
    gt_objects: Mapping[Any, Sequence[str]],
) -> dict[str, Any]:
    """Score object hallucination for a set of caption predictions.

    Args:
        predictions: Sequence of dicts, each with ``image_id``, ``caption`` and
            optionally ``references``.
        gt_objects: Mapping from image id (string, e.g. ``"139"``) to a list of
            COCO class names present in that image.  Names outside the 80 COCO
            classes are ignored.

    Returns:
        A JSON-serializable dict with:

        ``per_image``
            One record per prediction: ``image_id``, ``caption``, ``references``,
            ``mentioned`` (sorted unique canonical classes), ``hallucinated``
            (sorted unique classes absent from the ground truth), ``n_mentions``,
            ``n_hallucinated``, ``has_hallucination`` and ``reference_objects``
            (sorted unique canonical ground truth).
        ``chair_s``
            ``sum(n_hallucinated) / sum(n_mentions)``; ``0.0`` when nothing was
            mentioned.
        ``chair_i``
            Fraction of images with at least one hallucinated mention.
        ``n_images``, ``n_mentions``, ``n_hallucinated``
            Totals over the deduplicated per-caption mention sets.
        ``object_precision``
            Correctly mentioned classes / all mentioned classes (micro).
        ``object_recall``
            Correctly mentioned classes / all ground-truth classes, over images
            that have at least one ground-truth object (micro).
        ``object_f1``
            Harmonic mean of precision and recall.

    Notes:
        Mentions are deduplicated per caption, matching the standard CHAIR
        convention, so repeating a class within one caption cannot inflate the
        numerator or the denominator.
    """
    per_image: list[dict[str, Any]] = []

    total_mentions = 0
    total_hallucinated = 0
    total_correct = 0
    total_gt = 0
    images_with_hallucination = 0

    for prediction in predictions or []:
        image_id = prediction.get("image_id")
        caption = prediction.get("caption") or ""
        references = prediction.get("references") or []

        mentioned = sorted({mention["class_name"] for mention in extract_mentions(caption)})
        reference_objects = _normalize_gt(_lookup_gt(gt_objects, image_id))
        reference_set = set(reference_objects)

        hallucinated = sorted(name for name in mentioned if name not in reference_set)
        n_mentions = len(mentioned)
        n_hallucinated = len(hallucinated)
        has_hallucination = n_hallucinated > 0

        per_image.append(
            {
                "image_id": _to_native(image_id),
                "caption": caption,
                "references": list(references),
                "mentioned": mentioned,
                "hallucinated": hallucinated,
                "n_mentions": n_mentions,
                "n_hallucinated": n_hallucinated,
                "has_hallucination": has_hallucination,
                "reference_objects": reference_objects,
            }
        )

        total_mentions += n_mentions
        total_hallucinated += n_hallucinated
        total_correct += sum(1 for name in mentioned if name in reference_set)
        total_gt += len(reference_objects)
        images_with_hallucination += int(has_hallucination)

    n_images = len(per_image)
    chair_s = (total_hallucinated / total_mentions) if total_mentions else 0.0
    chair_i = (images_with_hallucination / n_images) if n_images else 0.0
    precision = (total_correct / total_mentions) if total_mentions else 0.0
    recall = (total_correct / total_gt) if total_gt else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "per_image": per_image,
        "chair_s": float(chair_s),
        "chair_i": float(chair_i),
        "n_images": int(n_images),
        "n_mentions": int(total_mentions),
        "n_hallucinated": int(total_hallucinated),
        "object_precision": float(precision),
        "object_recall": float(recall),
        "object_f1": float(f1),
    }
