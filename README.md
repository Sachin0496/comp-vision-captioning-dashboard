# Do Better Captions Hallucinate Less?

A controlled CNN–LSTM image-captioning study on COCO that measures **object
hallucination**, not just n-gram overlap. Three otherwise-matched captioners are
trained and evaluated:

| Model | Visual input | Attention | Detected objects |
|---|---|---|---|
| `global` | ResNet-18 global feature | ✗ | ✗ |
| `attention` | ResNet-18 7×7 spatial grid | ✓ (Bahdanau) | ✗ |
| `object` | ResNet-18 7×7 spatial grid | ✓ | ✓ (Faster R-CNN labels) |

All models are evaluated with BLEU-1–4, CIDEr-D, and CHAIR-style hallucination
rates. The report asks the central question directly: **does higher BLEU mean
fewer hallucinated objects?**

## Headline result

On 500 held-out COCO images:

| Model | BLEU-4 | CIDEr-D | CHAIRs | CHAIRi |
|---|---:|---:|---:|---:|
| Global CNN–LSTM | 11.94 | 0.485 | 49.32% | 62.00% |
| + attention | 16.32 | 0.704 | 34.01% | 40.20% |
| + detected object labels | **18.03** | **0.815** | **23.64%** | **27.60%** |

The object-aware model led on both caption quality and grounding. That does not
mean BLEU is sufficient: the pooled caption-level Spearman correlation between
sentence BLEU-4 and hallucinated-object count is only **−0.33**, and inside
matched BLEU bins the models still differ by up to ~33 percentage points in
hallucination rate. See [`report/report.pdf`](report/report.pdf) and
[`outputs/metrics/model_comparison.csv`](outputs/metrics/model_comparison.csv).

## Dataset

COCO 2017 validation images (5,000 images, 5 captions each, 80 object classes).
Because the full COCO train2017 archive is 18 GB, the 5k val split is
re-partitioned deterministically into:

- 4,000 train images
- 500 validation images
- 500 held-out test images

The detector is COCO-pretrained and therefore has not seen these images during
its own training. The caption models do see the 4k training images. This is a
deliberate laptop-scale compromise and is discussed in the report's limitations.

## Live dashboard

GitHub Pages serves the offline dashboard from the `dashboard/` folder at:

**https://sachin0496.github.io/comp-vision-captioning-dashboard/**

The deployment workflow is [`.github/workflows/pages.yml`](.github/workflows/pages.yml).
It publishes only `dashboard/`, so the site stays small and static.

## Interactive dashboard

A static, offline dashboard presents every assignment task, the dataset, model comparison,
hallucination examples, matched-BLEU analysis, attention explorer and downloads.

```bash
open "dashboard/index.html"          # no server needed
python scripts/build_dashboard.py    # regenerate payload + assets
```

See [`dashboard/README.md`](dashboard/README.md).

## Quick start

```bash
cd "/Users/sachin/Development/comp vision"

# 1. Environment (torch/torchvision already present on this machine)
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
export TORCH_HOME="$PWD/.cache/torch"
pip install -r requirements.txt

# 2. Data (COCO val2017 images + annotations, ~1.1 GB download)
bash scripts/00_download_data.sh
python scripts/01_prepare_data.py --force

# 3. One-time feature extraction (~60 s on an M4 MacBook Air)
python scripts/02_extract_features.py

# 4. Detected object labels (~10 min on MPS, memory-capped)
python scripts/03_detect_objects.py

# 5. Train all three models (~5 min each on CPU)
python scripts/04_train.py --model global
python scripts/04_train.py --model attention
python scripts/04_train.py --model object

# 6. Evaluate, test paired differences, and make figures
python scripts/05_evaluate.py
python scripts/06_visualize.py
python scripts/07_make_report.py
```

Or run the whole pipeline (download → training → report) with:

```bash
bash scripts/run_all.sh
```

A full run on the reference machine (Apple M4, 16 GB, 10 cores) takes roughly
35–45 minutes after download, with peak process memory well under 2 GB. The MPS
allocator is capped at 55% of unified memory in `src/utils.py`.

## Repository layout

```
configs/default.yaml       experiment configuration
src/
  data.py                  COCO parsing, splits, feature store, datasets
  features.py              frozen ResNet-18 feature extraction
  detector.py              Faster R-CNN object-label extraction
  models.py                global / attention / object-aware captioners
  inference.py             greedy decoding and prediction records
  hallucination.py         COCO synonym matching + CHAIRs/CHAIRi
  metrics.py               BLEU, CIDEr-D, bootstrap, correlations
  visualize.py             attention grids, comparison and qualitative figures
scripts/
  00_download_data.sh      fetch COCO val2017 + annotations
  01_prepare_data.py       splits, vocabulary, GT objects
  02_extract_features.py   cache 7x7x512 features
  03_detect_objects.py     cache detector labels/scores/boxes
  04_train.py              train one model
  05_evaluate.py           test metrics + paired comparisons
  06_visualize.py          report figures
  07_make_report.py        build report/report.md, .docx, .pdf
tests/                     unit tests (metrics, models, visualizations)
outputs/                   checkpoints, predictions, metrics (generated)
report/                    report + figures (generated)
```

## Evaluation definitions

**CHAIRs** = hallucinated object mentions / all object mentions, with mentions
deduplicated per caption. **CHAIRi** = fraction of images with at least one
hallucinated object. A mention counts only if it maps to one of the 80 COCO
classes via a curated synonym table; other nouns (sky, grass, water) are
ignored. This is the standard CHAIR protocol, with the known caveat that COCO
instance annotations are incomplete.

BLEU uses SacreBLEU's 13a tokenizer. CIDEr-D is implemented from scratch,
including global document frequencies and the length penalty, to avoid a Java
dependency. Confidence intervals are percentile bootstraps over images; paired
model comparisons resample the same images for both models.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Notes for laptop safety

- CNN features are extracted once and memory-mapped; the LSTM never runs a CNN.
- The detector is the small MobileNetV3-320 Faster R-CNN, run in batches of 4
  with a 55% MPS memory cap.
- Training runs on CPU with small batches; attention is computed over only 49
  spatial locations.
- `num_workers=0` for feature extraction/detection avoids macOS fork + MPS
  issues; image loading is still fast enough for 5k images.

## Deliverables

- `report/report.pdf` — full technical report with results and figures
- `outputs/metrics/` — machine-readable metrics and statistical tests
- `outputs/predictions/` — generated captions with attention maps
- `report/figures/` — attention maps, BLEU/CHAIR comparison, qualitative examples
- Unit tests for the metric implementation and all three models
