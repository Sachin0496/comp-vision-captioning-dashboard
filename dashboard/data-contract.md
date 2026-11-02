# Dashboard data contract (v1)

The dashboard is a static offline site. `scripts/build_dashboard.py` writes:

- `dashboard/data/dashboard.json` — machine-readable payload
- `dashboard/data/dashboard-data.js` — same payload wrapped as
  `window.DASHBOARD_DATA = {...};` so `index.html` works from `file://`
- `dashboard/assets/images/<image_id>.jpg` — selected COCO images
- `dashboard/assets/attention/<image_id>_<word_index>.png` — overlays
- `dashboard/assets/figures/*.png` — copied report figures
- `dashboard/assets/downloads/*` — report/metrics downloads

`dashboard/app.js` must only read `window.DASHBOARD_DATA`. Do not use `fetch`
for the payload (it is blocked under `file://`).

## Top-level schema

```json
{
  "meta": {
    "title": "string",
    "subtitle": "string",
    "generated_at": "ISO-8601 string",
    "split": "test",
    "split_note": "string",
    "n_test": 500,
    "n_train": 4000,
    "n_val": 500,
    "n_total": 5000,
    "vocab_size": 2916,
    "train_captions": 20010,
    "test_captions": 2501,
    "seed": 42,
    "device_note": "string",
    "dataset_note": "string"
  },
  "headline": {
    "question": "string",
    "answer": "string",
    "best_model": "object",
    "best_bleu4": 18.03,
    "best_chair_s": 23.64,
    "pooled_rho": -0.327,
    "pooled_p": 8.1e-39,
    "cluster_rho": -0.327,
    "cluster_ci": [-0.386, -0.268],
    "cluster_p": 0.0,
    "matched_bin": "10-20",
    "matched_rates": {"global": 66.1, "attention": 43.1, "object": 32.5},
    "hallucination_reduction_pp": 25.68,
    "bleu_gain_points": 6.08
  },
  "tasks": [
    {
      "id": "dataset",
      "label": "Dataset with captions + object annotations",
      "description": "string",
      "status": "complete",
      "metric": "5,000 images · 80 classes",
      "evidence": [{"label": "prepare data", "path": "../scripts/01_prepare_data.py"}]
    }
  ],
  "pipeline": [
    {
      "id": "data",
      "label": "Dataset",
      "detail": "string",
      "status": "complete",
      "runtime": "string",
      "file": "../scripts/01_prepare_data.py"
    }
  ],
  "models": [
    {
      "id": "global",
      "label": "Global CNN-LSTM",
      "short": "Global",
      "color": "#4C72B0",
      "description": "string",
      "params": 4344676,
      "params_label": "4.34 M",
      "train_minutes": 6.9,
      "bleu1": 56.84,
      "bleu2": 35.74,
      "bleu3": 20.65,
      "bleu4": 11.94,
      "bleu": 11.94,
      "cider_d": 0.485,
      "chair_s": 49.32,
      "chair_i": 62.0,
      "object_precision": 50.68,
      "object_recall": 30.04,
      "object_f1": 37.72,
      "n_mentions": 809,
      "n_hallucinated": 399,
      "sentence_bleu4_mean": 0.0,
      "unique_captions": 0,
      "uses_attention": false,
      "uses_objects": false
    }
  ],
  "pairwise": [
    {
      "id": "global_vs_attention",
      "first": "global",
      "second": "attention",
      "label": "Global vs Attention",
      "objects_diff": 0.296,
      "objects_ci": [0.226, 0.368],
      "objects_p": 0.0,
      "rate_diff_pp": 21.8,
      "rate_ci_pp": [17.0, 26.8],
      "bleu4_diff": -3.267,
      "bleu4_ci": [-4.184, -2.411],
      "mcnemar_p": 8.4e-16,
      "only_first": 150,
      "only_second": 41
    }
  ],
  "correlations": {
    "global": {"n": 500, "rho": -0.326, "p": 4.6e-13, "rate": 62.0},
    "attention": {"n": 500, "rho": -0.284, "p": 1.2e-10, "rate": 40.2},
    "object": {"n": 500, "rho": -0.249, "p": 8.0e-8, "rate": 27.6},
    "pooled": {"n": 1500, "rho": -0.327, "p": 8.1e-39, "rate": 43.27}
  },
  "quartiles": [
    {
      "bucket": "Q1",
      "n": 375,
      "bleu_min": 4.41,
      "bleu_max": 10.55,
      "bleu_mean": 7.7,
      "rate": 62.67,
      "ci": [57.9, 66.9],
      "mean_hallucinated": 0.87
    }
  ],
  "matched_bleu": [
    {
      "bin": "0-10",
      "model": "global",
      "n": 127,
      "mean_bleu4": 7.0,
      "rate": 77.2,
      "ci": [69.3, 84.3]
    }
  ],
  "training": {
    "global": [
      {"epoch": 1, "train_loss": 6.02, "val_loss": 4.52, "val_chair_s": 68.0, "val_bleu4": 0.02}
    ]
  },
  "dataset": {
    "splits": {"train": 4000, "val": 500, "test": 500},
    "captions": {"train": 20010, "val": 2503, "test": 2501},
    "caption_length": {"mean": 10.45, "p50": 10, "p95": 14, "max": 39},
    "objects_per_image": {"mean": 2.93, "max": 14},
    "top_objects": [{"name": "person", "count": 4000}],
    "detector": {
      "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
      "threshold": 0.3,
      "max_detections": 12,
      "mean_detections": 4.81,
      "empty_images": 62,
      "seconds": 782.9,
      "images": 5000
    }
  },
  "hallucination": {
    "top_hallucinated": [{"name": "person", "count": 120}],
    "examples": [
      {
        "image_id": 724,
        "image": "assets/images/724.jpg",
        "references": ["string"],
        "gt_objects": ["car", "stop sign", "truck"],
        "captions": {
          "global": {"text": "string", "hallucinated": ["clock"]},
          "attention": {"text": "string", "hallucinated": []},
          "object": {"text": "string", "hallucinated": []}
        }
      }
    ]
  },
  "attention_examples": [
    {
      "image_id": 724,
      "image": "assets/images/724.jpg",
      "caption": "string",
      "words": ["a", "stop", "sign"],
      "overlays": [
        {"index": 1, "word": "stop", "path": "assets/attention/724_1.png"}
      ],
      "highlight": [1, 2]
    }
  ],
  "gallery": [
    {
      "image_id": 724,
      "image": "assets/images/724.jpg",
      "references": ["string"],
      "gt_objects": ["car"],
      "captions": {
        "global": {"text": "string", "hallucinated": ["clock"]},
        "attention": {"text": "string", "hallucinated": []},
        "object": {"text": "string", "hallucinated": []}
      }
    }
  ],
  "figures": [
    {
      "id": "model_comparison",
      "label": "Model comparison",
      "path": "assets/figures/model_comparison.png",
      "caption": "string"
    }
  ],
  "artifacts": [
    {
      "label": "MAD-format report (DOCX)",
      "path": "assets/downloads/MAD_Report_Image_Captioning.docx",
      "kind": "docx",
      "size": 123456
    }
  ]
}
```

## Required semantics

- Percentages are 0–100; `cider_d` is 0–10.
- `models` order: `global`, `attention`, `object`.
- `pairwise` keys: `global_vs_attention`, `global_vs_object`,
  `attention_vs_object`. `objects_diff` is first minus second (positive means
  first hallucinates more).
- `matched_bleu` has 5 bins × 3 models, ordered bin then model, with bins
  `0-10`, `10-20`, `20-30`, `30-40`, `40+`.
- `tasks` must cover the assignment: dataset, architectures, BLEU, CHAIR,
  attention visualisation, BLEU↔hallucination analysis, report.
- Every path is relative to `dashboard/` and must exist.
- JSON must be strict (`allow_nan=False`).
- Select 8 gallery images, 4 attention examples, 6 hallucination examples;
  deterministic ordering (sort by image id as tie-breaker).
- Attention overlays: render a 7×7 attention map onto the image, normalized
  robustly, with an inferno/turbo colormap and ~50% alpha. Save PNG at image
  width (max 900 px).
- Images: resize to max width 900 px, JPEG quality 82, RGB.

## UI requirements

`dashboard/index.html`, `dashboard/styles.css`, `dashboard/app.js`:

- Hero/overview with the research question, answer, 4 KPI cards, and model
  comparison bars.
- Sticky section nav: Overview, Pipeline, Dataset, Models, Hallucination,
  BLEU vs CHAIR, Attention, Gallery, Report.
- Pipeline/task coverage cards with status and evidence links.
- Dataset panel: split bars, caption stats, top COCO objects, detector stats.
- Models panel: cards + full metrics table + interactive BLEU/CHAIR bars.
- Hallucination panel: CHAIR cards, top hallucinated objects, example cards
  with hallucinated words highlighted in red.
- BLEU vs CHAIR panel: correlation cards, quartile table, matched-BLEU chart,
  and the scatter figure.
- Attention explorer: example selector, word chips, image/overlay toggle,
  caption display.
- Gallery: 8 example cards, filter by model (show captions) and text search.
- Report panel: download cards for DOCX/PDF/CSV, plus reproducibility notes.
- Lightbox for figures. "Print / Save PDF" button. Responsive down to 360 px.
- No external CDN/fonts/libraries; system font stack; vanilla JS only.
- Handle missing/partial data gracefully with a visible error state.
- Accessible: buttons, aria labels, keyboard focus styles, `prefers-reduced-motion`.
- Preserve `window.DASHBOARD_DATA`; do not mutate the payload.
