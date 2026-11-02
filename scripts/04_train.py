#!/usr/bin/env python
"""Train one of the three captioners on precomputed CNN features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    CaptionDataset,
    GenerationDataset,
    collate_captions,
    load_processed_bundle,
)
from src.inference import generate_predictions  # noqa: E402
from src.metrics import evaluate_predictions  # noqa: E402
from src.models import build_model, save_checkpoint  # noqa: E402
from src.utils import (  # noqa: E402
    configure_runtime,
    ensure_dir,
    load_config,
    save_json,
    seed_everything,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["global", "attention", "object"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-eval", action="store_true", help="Skip per-epoch BLEU/CHAIR evaluation")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    for key in ("tokens", "spatial", "global", "obj_labels", "obj_scores", "obj_mask"):
        if key in batch:
            batch[key] = batch[key].to(device)
    return batch


def masked_loss(logits: torch.Tensor, tokens: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    targets = tokens[:, 1:]
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch["tokens"],
            batch["spatial"],
            batch["global"],
            batch["obj_labels"],
            batch["obj_scores"],
            batch["obj_mask"],
        )
        loss = masked_loss(output["logits"], batch["tokens"], criterion)
        counts = int((batch["tokens"][:, 1:] != 0).sum())
        total_loss += float(loss) * counts
        total_tokens += counts
    return total_loss / max(1, total_tokens)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    epochs = int(args.epochs or config["train"]["epochs"])
    batch_size = int(args.batch_size or config["train"]["batch_size"])
    lr = float(args.lr or config["train"]["lr"])
    device = configure_runtime(
        args.device,
        mps_memory_fraction=float(config["runtime"]["mps_memory_fraction"]),
        torch_threads=int(config["runtime"]["torch_threads"]),
    )
    seed_everything(int(config["seed"]))
    run_name = args.model
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "checkpoints" / run_name)
    setup_logging(log_file=output_dir / "train.log")
    log = __import__("logging").getLogger(__name__)

    bundle = load_processed_bundle(config)
    splits = bundle["splits"]
    train_ids = splits["train"][: args.limit_train] if args.limit_train else splits["train"]
    val_ids = splits["val"][: args.limit_val] if args.limit_val else splits["val"]
    if args.model == "object" and bundle["detections"] is None:
        raise FileNotFoundError("Object-aware training needs data/processed/detections.json")
    name_to_idx = bundle["categories"]["name_to_idx"] if args.model == "object" else {}
    detections = bundle["detections"] if args.model == "object" else None
    max_len = int(config["data"]["max_caption_len"])
    samples_per_epoch = int(config["train"].get("samples_per_epoch", 0) or 0)
    val_captions_per_image = int(config["train"].get("val_captions_per_image", 1))

    train_dataset = CaptionDataset(
        train_ids,
        bundle["captions"],
        bundle["features"],
        bundle["vocab"],
        detections=detections,
        name_to_idx=name_to_idx,
        captions_per_image=None,
        max_len=max_len,
        max_objects=int(config["detector"]["max_detections"]),
        training=True,
    )
    val_loss_dataset = CaptionDataset(
        val_ids,
        bundle["captions"],
        bundle["features"],
        bundle["vocab"],
        detections=detections,
        name_to_idx=name_to_idx,
        captions_per_image=val_captions_per_image,
        max_len=max_len,
        max_objects=int(config["detector"]["max_detections"]),
    )
    val_gen_dataset = GenerationDataset(
        val_ids,
        bundle["captions"],
        bundle["features"],
        detections=detections,
        name_to_idx=name_to_idx,
        max_objects=int(config["detector"]["max_detections"]),
    )
    effective_samples = min(len(train_dataset), samples_per_epoch) if samples_per_epoch else len(train_dataset)
    sampler = (
        RandomSampler(train_dataset, num_samples=effective_samples, replacement=False)
        if effective_samples < len(train_dataset)
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collate_captions,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_loss_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_captions,
    )

    model = build_model(
        args.model,
        vocab_size=len(bundle["vocab"]),
        config=config,
        spatial_dim=int(bundle["features"].meta["spatial_dim"]),
        num_object_classes=len(bundle["categories"]["names"]),
    ).to(device)
    log.info(
        "Training %s on %d captions (%d images), %d sampled/epoch, val=%d images, device=%s, params=%.2fM",
        args.model,
        len(train_dataset),
        len(train_ids),
        effective_samples,
        len(val_ids),
        device,
        model.count_parameters() / 1e6,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=float(config["train"]["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max(1, epochs * len(train_loader)),
        pct_start=0.08,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    history: list[dict] = []
    best_score = float("-inf")
    best_epoch = 0
    patience = int(config["train"]["patience"])
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["tokens"],
                batch["spatial"],
                batch["global"],
                batch["obj_labels"],
                batch["obj_scores"],
                batch["obj_mask"],
            )
            loss = masked_loss(output["logits"], batch["tokens"], criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            optimizer.step()
            scheduler.step()

            counts = int((batch["tokens"][:, 1:] != 0).sum())
            running_loss += float(loss.detach()) * counts
            running_tokens += counts
            if step % 50 == 0 or step == len(train_loader):
                log.info(
                    "epoch %d/%d step %d/%d lr=%.2e loss=%.4f",
                    epoch,
                    epochs,
                    step,
                    len(train_loader),
                    scheduler.get_last_lr()[0],
                    running_loss / max(1, running_tokens),
                )

        train_loss = running_loss / max(1, running_tokens)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": round(time.perf_counter() - epoch_start, 2),
        }

        if not args.no_eval:
            predictions = generate_predictions(
                model,
                val_gen_dataset,
                bundle["vocab"],
                device=device,
                batch_size=64,
                max_len=max_len,
                return_attention=False,
            )
            metrics = evaluate_predictions(predictions, bundle["gt_objects"])
            record.update(
                {
                    "val_bleu1": metrics["bleu1"],
                    "val_bleu2": metrics["bleu2"],
                    "val_bleu3": metrics["bleu3"],
                    "val_bleu4": metrics["bleu4"],
                    "val_cider_d": metrics["cider_d"],
                    "val_chair_s": metrics["chair_s"],
                    "val_chair_i": metrics["chair_i"],
                }
            )
            score = -val_loss
            log.info(
                "epoch %d: train=%.4f val=%.4f BLEU-4=%.2f CIDEr-D=%.3f CHAIRs=%.2f%% CHAIRi=%.2f%% (%.1fs)",
                epoch,
                train_loss,
                val_loss,
                metrics["bleu4"],
                metrics["cider_d"],
                100 * metrics["chair_s"],
                100 * metrics["chair_i"],
                record["seconds"],
            )
        else:
            score = -val_loss
            log.info("epoch %d: train=%.4f val=%.4f (%.1fs)", epoch, train_loss, val_loss, record["seconds"])

        history.append(record)
        save_json(history, output_dir / "history.json")
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(output_dir / "best.pt", model, args.model, bundle["vocab"], config, epoch, record)
        else:
            stale += 1
        save_checkpoint(output_dir / "last.pt", model, args.model, bundle["vocab"], config, epoch, record)
        if stale >= patience:
            log.info("Early stopping after %d epochs without improvement (best epoch %d)", stale, best_epoch)
            break

    log.info(
        "Finished %s in %.1f min; best epoch %d; checkpoint at %s",
        args.model,
        (time.perf_counter() - started) / 60.0,
        best_epoch,
        output_dir / "best.pt",
    )
    summary = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_run": len(history),
        "history": history,
        "train_captions": len(train_dataset),
        "val_images": len(val_ids),
        "parameters": model.count_parameters(),
        "device": str(device),
    }
    save_json(summary, output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
