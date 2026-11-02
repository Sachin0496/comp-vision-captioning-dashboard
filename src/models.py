"""CNN-LSTM captioners: global, spatial attention, and object-aware fusion."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .text import Vocab


class ObjectEncoder(nn.Module):
    """Score-weighted embedding of detected object labels."""

    def __init__(self, num_classes: int, embed_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        # Index 0 is reserved for padding / "no object"; real labels are 1..C.
        self.embedding = nn.Embedding(num_classes + 1, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, labels: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embedding(labels))
        weights = scores * mask.to(scores.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=1) / denominator


class Captioner(nn.Module):
    """An LSTM caption decoder with optional visual attention and object fusion.

    ``use_attention=False`` is the Show-and-Tell-style baseline: the global CNN
    feature initialises the LSTM state and is then discarded.  With attention,
    the decoder attends over the 7x7 spatial grid at every time step.  Adding
    ``obj_dim`` concatenates a score-weighted detector-label embedding to each
    LSTM input.
    """

    def __init__(
        self,
        vocab_size: int,
        spatial_dim: int = 512,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        attn_dim: int = 512,
        dropout: float = 0.3,
        use_attention: bool = True,
        obj_dim: int = 0,
        num_object_classes: int = 80,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.spatial_dim = int(spatial_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_attention = bool(use_attention)
        self.obj_dim = int(obj_dim)
        self.num_object_classes = int(num_object_classes)

        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.global_to_h = nn.Linear(spatial_dim, hidden_dim)
        self.global_to_c = nn.Linear(spatial_dim, hidden_dim)

        if self.use_attention:
            self.spatial_proj = nn.Linear(spatial_dim, attn_dim)
            self.att_h = nn.Linear(hidden_dim, attn_dim)
            self.att_v = nn.Linear(attn_dim, 1)

        if self.obj_dim > 0:
            self.object_encoder = ObjectEncoder(num_object_classes, self.obj_dim, dropout=dropout)
        else:
            self.object_encoder = None

        input_dim = embed_dim
        if self.use_attention:
            input_dim += spatial_dim
        if self.obj_dim > 0:
            input_dim += self.obj_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    # State / feature handling
    # ------------------------------------------------------------------
    def init_state(self, global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.global_to_h(global_features))
        cell = torch.tanh(self.global_to_c(global_features))
        return hidden, cell

    def encode_objects(
        self,
        labels: torch.Tensor | None,
        scores: torch.Tensor | None,
        mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.object_encoder is None:
            return None
        if labels is None or labels.numel() == 0:
            return torch.zeros(batch_size, self.obj_dim, device=device)
        if scores is None:
            scores = torch.ones_like(labels, dtype=torch.float32)
        if mask is None:
            mask = labels > 0
        return self.object_encoder(labels, scores, mask)

    def step(
        self,
        word_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        spatial: torch.Tensor,
        obj_vector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        """One decoder step.  Returns logits, new state, attention weights."""
        hidden, cell = state
        embedded = self.word_embedding(word_ids)
        inputs = [embedded]
        attention: torch.Tensor | None = None
        if self.use_attention:
            projected = self.spatial_proj(spatial)  # [B, 49, A]
            energy = self.att_v(torch.tanh(projected + self.att_h(hidden).unsqueeze(1))).squeeze(-1)
            attention = torch.softmax(energy, dim=1)  # [B, 49]
            context = torch.bmm(attention.unsqueeze(1), spatial).squeeze(1)  # [B, D]
            inputs.append(context)
        if obj_vector is not None:
            inputs.append(obj_vector)
        lstm_input = torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        logits = self.classifier(self.dropout(hidden))
        return logits, (hidden, cell), attention

    # ------------------------------------------------------------------
    # Teacher-forced training / scoring
    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        spatial: torch.Tensor,
        global_features: torch.Tensor,
        obj_labels: torch.Tensor | None = None,
        obj_scores: torch.Tensor | None = None,
        obj_mask: torch.Tensor | None = None,
    ) -> dict:
        """Compute next-token logits for ``tokens[:, :-1]``."""
        if tokens.size(1) < 2:
            raise ValueError("Caption sequences must include BOS and EOS")
        state = self.init_state(global_features)
        obj_vector = self.encode_objects(
            obj_labels, obj_scores, obj_mask, tokens.size(0), tokens.device
        )
        logits: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        for timestep in range(tokens.size(1) - 1):
            step_logits, state, attention = self.step(tokens[:, timestep], state, spatial, obj_vector)
            logits.append(step_logits)
            if attention is not None:
                attentions.append(attention)
        return {
            "logits": torch.stack(logits, dim=1),
            "attention": torch.stack(attentions, dim=1) if attentions else None,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        spatial: torch.Tensor,
        global_features: torch.Tensor,
        obj_labels: torch.Tensor | None = None,
        obj_scores: torch.Tensor | None = None,
        obj_mask: torch.Tensor | None = None,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        unk_id: int = 3,
        max_len: int = 22,
        return_attention: bool = True,
    ) -> dict:
        """Greedy decode a batch of images, returning token ids and attention."""
        batch_size = spatial.size(0)
        device = spatial.device
        state = self.init_state(global_features)
        obj_vector = self.encode_objects(obj_labels, obj_scores, obj_mask, batch_size, device)
        words = torch.full((batch_size,), int(bos_id), dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        output_ids: list[torch.Tensor] = []
        output_attention: list[torch.Tensor] = []
        for _ in range(int(max_len)):
            logits, state, attention = self.step(words, state, spatial, obj_vector)
            # Never emit padding/BOS/UNK as caption words; EOS remains allowed.
            blocked = torch.tensor([pad_id, bos_id, unk_id], device=logits.device, dtype=torch.long)
            logits = logits.clone()
            logits[:, blocked] = float("-inf")
            next_words = logits.argmax(dim=-1)
            next_words = torch.where(finished, torch.full_like(next_words, int(pad_id)), next_words)
            output_ids.append(next_words)
            if return_attention and attention is not None:
                output_attention.append(attention)
            finished = finished | (next_words == int(eos_id))
            words = next_words
            if bool(finished.all()):
                break
        ids = torch.stack(output_ids, dim=1)  # [B, L]
        attention_tensor = torch.stack(output_attention, dim=1) if output_attention else None
        return {"ids": ids, "attention": attention_tensor}

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_model(
    model_name: str,
    vocab_size: int,
    config: dict,
    spatial_dim: int = 512,
    num_object_classes: int = 80,
) -> Captioner:
    """Instantiate one of the three controlled variants."""
    if model_name not in {"global", "attention", "object"}:
        raise ValueError(f"Unknown model {model_name!r}")
    model_cfg = config["model"]
    use_attention = model_name in {"attention", "object"}
    obj_dim = int(model_cfg["obj_dim"]) if model_name == "object" else 0
    return Captioner(
        vocab_size=vocab_size,
        spatial_dim=spatial_dim,
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        attn_dim=int(model_cfg.get("attn_dim", model_cfg["hidden_dim"])),
        dropout=float(model_cfg["dropout"]),
        use_attention=use_attention,
        obj_dim=obj_dim,
        num_object_classes=num_object_classes,
    )


def save_checkpoint(
    path: str | Path,
    model: Captioner,
    model_name: str,
    vocab: Vocab,
    config: dict,
    epoch: int,
    metrics: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "model_kwargs": {
            "vocab_size": model.vocab_size,
            "spatial_dim": model.spatial_dim,
            "embed_dim": model.embed_dim,
            "hidden_dim": model.hidden_dim,
            "attn_dim": model.spatial_proj.out_features if model.use_attention else model.hidden_dim,
            "dropout": model.dropout.p,
            "use_attention": model.use_attention,
            "obj_dim": model.obj_dim,
            "num_object_classes": model.num_object_classes,
        },
        "state_dict": model.state_dict(),
        "vocab": vocab.to_dict(),
        "config": config,
        "epoch": int(epoch),
        "metrics": metrics or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[Captioner, Vocab, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = Captioner(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    vocab = Vocab.from_dict(payload["vocab"])
    return model, vocab, payload
