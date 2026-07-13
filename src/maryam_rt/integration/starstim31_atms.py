from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from maryam_rt.hierarchical.models.loss import ClipLoss
from maryam_rt.hierarchical.models.subject_layers.Embed import DataEmbedding


@dataclass(frozen=True)
class Starstim31ATMSConfig:
    seq_len: int = 250
    pred_len: int = 250
    d_model: int = 250
    enc_in: int = 31
    task_name: str = "classification"
    output_attention: bool = False
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.25
    factor: int = 1
    n_heads: int = 4
    e_layers: int = 8
    d_ff: int = 256
    activation: str = "gelu"


class _FlattenProjection(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, x.size(1))


class _FlattenHead(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous().view(x.size(0), -1)


class _ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return x + self.fn(x, **kwargs)


class _FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1, output_attention: bool = False) -> None:
        super().__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor | None]:
        _, _, _, channels = queries.shape
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scores / sqrt(channels), dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attention, values)
        if self.output_attention:
            return values.contiguous(), attention
        return values.contiguous(), None


class _AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attention_dropout: float, output_attention: bool) -> None:
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = _FullAttention(
            attention_dropout=attention_dropout,
            output_attention=output_attention,
        )
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor | None]:
        batch, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        heads = self.n_heads
        queries = self.query_projection(queries).view(batch, query_len, heads, -1)
        keys = self.key_projection(keys).view(batch, key_len, heads, -1)
        values = self.value_projection(values).view(batch, key_len, heads, -1)
        out, attn = self.inner_attention(queries, keys, values)
        out = out.view(batch, query_len, -1)
        return self.out_projection(out), attn


class _EncoderLayer(nn.Module):
    def __init__(self, attention: _AttentionLayer, d_model: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class _Encoder(nn.Module):
    def __init__(self, attn_layers: list[_EncoderLayer], norm_layer: nn.Module) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = None
        self.norm = norm_layer

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor | None]]:
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)
        return self.norm(x), attns


class _Starstim31ITransformer(nn.Module):
    def __init__(self, config: Starstim31ATMSConfig, num_subjects: int) -> None:
        super().__init__()
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.output_attention = config.output_attention
        self.enc_in = config.enc_in
        self.enc_embedding = DataEmbedding(
            config.seq_len,
            config.d_model,
            config.embed,
            config.freq,
            config.dropout,
            joint_train=False,
            num_subjects=num_subjects,
        )
        self.encoder = _Encoder(
            [
                _EncoderLayer(
                    _AttentionLayer(
                        config.d_model,
                        config.n_heads,
                        attention_dropout=config.dropout,
                        output_attention=config.output_attention,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model),
        )

    def forward(self, x_enc: Tensor, subject_ids: Tensor) -> Tensor:
        enc_out = self.enc_embedding(x_enc, None, subject_ids)
        enc_out, _ = self.encoder(enc_out)
        if enc_out.shape[1] > self.enc_in:
            return enc_out[:, 1 : self.enc_in + 1, :]
        return enc_out[:, : self.enc_in, :]


class _PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40, num_channels: int = 31) -> None:
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            _FlattenProjection(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.tsconv(x.unsqueeze(1)))


class _EncEEGATMS(nn.Sequential):
    def __init__(self, emb_size: int = 40, num_channels: int = 31) -> None:
        super().__init__(
            _PatchEmbedding(emb_size=emb_size, num_channels=num_channels),
            _FlattenHead(),
        )


class _ProjEEGATMS(nn.Sequential):
    def __init__(self, embedding_dim: int = 1440, proj_dim: int = 1024, drop_proj: float = 0.5) -> None:
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            _ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class Starstim31ATMS(nn.Module):
    def __init__(
        self,
        *,
        num_channels: int = 31,
        sequence_length: int = 250,
        num_subjects: int = 11,
        embedding_dim: int = 1440,
        transformer_layers: int = 8,
        transformer_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_subjects = num_subjects
        self.config = Starstim31ATMSConfig(
            seq_len=sequence_length,
            pred_len=sequence_length,
            d_model=sequence_length,
            enc_in=num_channels,
            dropout=transformer_dropout,
            e_layers=transformer_layers,
        )
        self.encoder = _Starstim31ITransformer(self.config, num_subjects=num_subjects)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(self.config.d_model, sequence_length) for _ in range(num_subjects)]
        )
        self.enc_eeg = _EncEEGATMS(emb_size=40, num_channels=num_channels)
        self.proj_eeg = _ProjEEGATMS(embedding_dim=embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x: Tensor, subject_ids: Tensor) -> Tensor:
        x = self.encoder(x, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        return self.proj_eeg(eeg_embedding)


class Starstim31ATMSEmbedder:
    """Load the Starstim31 ATMS checkpoint and export EEG CLIP-space embeddings."""

    def __init__(self, checkpoint_path: Path, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Starstim31 ATMS checkpoint not found: {self.checkpoint_path}")
        self.device_name = device
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        num_subjects = int(state["encoder.enc_embedding.subject_embedding.subject_embedding.weight"].shape[0])
        layer_indices = {
            int(key.split(".")[3])
            for key in state
            if key.startswith("encoder.encoder.attn_layers.") and key.split(".")[3].isdigit()
        }
        transformer_layers = max(layer_indices) + 1
        embedding_dim = int(state["proj_eeg.0.weight"].shape[1])
        self.model = Starstim31ATMS(
            num_subjects=num_subjects,
            embedding_dim=embedding_dim,
            transformer_layers=transformer_layers,
        ).to(device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.num_subjects = num_subjects

    @torch.inference_mode()
    def embed(self, epoch_lab31: np.ndarray | Tensor, subject_id: int | None) -> Tensor:
        """Embed a known subject, or use the checkpoint's trained shared token.

        ``subject_id=None`` (or the explicit sentinel ``num_subjects``) selects
        ``SubjectEmbedding.shared_embedding``.  This is the only non-arbitrary
        conditioning available for a participant who was not assigned a
        checkpoint subject row.
        """
        resolved_subject_id = self.num_subjects if subject_id is None else int(subject_id)
        if resolved_subject_id < 0 or resolved_subject_id > self.num_subjects:
            raise ValueError(
                f"Starstim31 ATMS subject_id {resolved_subject_id} is outside known range "
                f"0..{self.num_subjects - 1} and shared sentinel {self.num_subjects}."
            )
        if isinstance(epoch_lab31, np.ndarray):
            x = torch.from_numpy(np.asarray(epoch_lab31, dtype=np.float32))
        else:
            x = epoch_lab31.detach().float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if tuple(x.shape[1:]) != (31, 250):
            raise ValueError(f"Expected Starstim31 epoch shaped (B, 31, 250), got {tuple(x.shape)}.")
        subject_ids = torch.full(
            (x.shape[0],),
            resolved_subject_id,
            dtype=torch.long,
            device=self.device_name,
        )
        return self.model(x.to(self.device_name), subject_ids).detach().cpu()

    def save_embedding(
        self,
        epoch_lab31: np.ndarray | Tensor,
        subject_id: int | None,
        output_path: Path,
    ) -> dict[str, Any]:
        embedding = self.embed(epoch_lab31, subject_id=subject_id)
        resolved_subject_id = self.num_subjects if subject_id is None else int(subject_id)
        subject_mode = "shared_unseen" if resolved_subject_id == self.num_subjects else "known"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding": embedding,
            "subject_id": resolved_subject_id,
            "subject_mode": subject_mode,
            "checkpoint_path": str(self.checkpoint_path),
            "model": "starstim31_atms",
        }
        torch.save(payload, output_path)
        return {
            "path": str(output_path),
            "shape": list(embedding.shape),
            "subject_id": resolved_subject_id,
            "subject_mode": subject_mode,
            "checkpoint_path": str(self.checkpoint_path),
        }
