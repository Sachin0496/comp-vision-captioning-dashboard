"""Batch caption generation and prediction serialisation."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .data import collate_generation
from .models import Captioner
from .text import Vocab, detokenize


def _trim_ids(ids: Sequence[int], eos_id: int) -> list[int]:
    output: list[int] = []
    for token_id in ids:
        if int(token_id) == int(eos_id):
            break
        output.append(int(token_id))
    return output


@torch.inference_mode()
def generate_predictions(
    model: Captioner,
    dataset: Dataset,
    vocab: Vocab,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    max_len: int = 22,
    return_attention: bool = False,
    progress: bool = False,
) -> list[dict]:
    """Generate one caption per image, preserving the dataset order."""
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_generation,
    )
    model.eval().to(device)
    predictions: list[dict] = []
    iterator: Iterable = loader
    if progress:
        from tqdm import tqdm

        iterator = tqdm(loader, desc="generate", leave=False)
    for batch in iterator:
        spatial = batch["spatial"].to(device)
        global_features = batch["global"].to(device)
        obj_labels = batch["obj_labels"].to(device)
        obj_scores = batch["obj_scores"].to(device)
        obj_mask = batch["obj_mask"].to(device)
        output = model.generate(
            spatial,
            global_features,
            obj_labels=obj_labels,
            obj_scores=obj_scores,
            obj_mask=obj_mask,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id,
            unk_id=vocab.unk_id,
            max_len=max_len,
            return_attention=return_attention,
        )
        ids = output["ids"].detach().cpu()
        attention = output["attention"].detach().cpu() if output["attention"] is not None else None
        for row, image_id in enumerate(batch["image_id"].tolist()):
            token_ids = _trim_ids(ids[row].tolist(), vocab.eos_id)
            words = [vocab.itos[token] for token in token_ids if 0 <= token < len(vocab.itos)]
            record = {
                "image_id": int(image_id),
                "caption": detokenize(words),
                "references": list(batch["references"][row]),
            }
            if attention is not None and return_attention:
                count = min(len(words), attention.size(1))
                record["words"] = words[:count]
                flat_maps = attention[row, :count]
                side = int(round(flat_maps.size(-1) ** 0.5))
                if side * side == flat_maps.size(-1):
                    flat_maps = flat_maps.view(count, side, side)
                record["attention"] = [flat_maps[index].tolist() for index in range(count)]
            predictions.append(record)
    return predictions
