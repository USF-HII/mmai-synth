# =============================================
# FILE: mmai/synth/fastq_transformer.py 
# =============================================
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DNA_VOCAB = ["A","C","G","T","N"]
PAD = "<pad>"; BOS = "<bos>"; EOS = "<eos>"
ALL_TOKENS = [PAD, BOS, EOS] + DNA_VOCAB
stoi = {ch:i for i,ch in enumerate(ALL_TOKENS)}
itos = {i:ch for ch,i in stoi.items()}


def encode(seq: str):
    return [stoi[BOS]] + [stoi.get(ch, stoi["N"]) for ch in seq.upper()] + [stoi[EOS]]

def decode(ids: List[int]):
    # Drop BOS, stop at EOS
    out = []
    for i in ids:
        ch = itos.get(i, "N")
        if ch == EOS: break
        if ch in (BOS, PAD): continue
        out.append(ch)
    return "".join(out)


@dataclass
class FASTQTransformerConfig:
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    max_len: int = 512
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 5
    seed: int = 42
    device: str = "auto"  # "cpu" | "cuda" | "auto"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CharTransformer(nn.Module):
    def __init__(self, cfg: FASTQTransformerConfig):
        super().__init__()
        self.cfg = cfg
        vocab_size = len(ALL_TOKENS)
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_head, batch_first=True, dim_feedforward=4*cfg.d_model, dropout=cfg.dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layer)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, x):
        # x: (B, T)
        h = self.embed(x)
        h = self.pos(h)
        h = self.encoder(h)
        return self.lm_head(h)


def _prepare_batches(seqs: List[str], max_len: int, batch_size: int, device: torch.device):
    # Filter overly long
    enc = [encode(s)[:max_len] for s in seqs]
    # Create (input, target) by shifting
    batches = []
    i = 0
    while i < len(enc):
        chunk = enc[i:i+batch_size]
        T = max(len(s) for s in chunk)
        pad_id = stoi[PAD]
        x = torch.full((len(chunk), T-1), pad_id, dtype=torch.long)
        y = torch.full((len(chunk), T-1), pad_id, dtype=torch.long)
        for b, s in enumerate(chunk):
            # ensure at least length 2 (BOS + EOS)
            if len(s) < 2:
                s = [stoi[BOS], stoi[EOS]]
            x[b, :len(s)-1] = torch.tensor(s[:-1], dtype=torch.long)
            y[b, :len(s)-1] = torch.tensor(s[1:], dtype=torch.long)
        batches.append((x.to(device), y.to(device)))
        i += batch_size
    return batches


def train_transformer(seqs: List[str], cfg: FASTQTransformerConfig) -> CharTransformer:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    model = CharTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    batches = _prepare_batches(seqs, cfg.max_len, cfg.batch_size, device)
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for x, y in batches:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=stoi[PAD])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
    return model


def sample_sequences(model: CharTransformer, n: int, length_sampler, max_len: int, temperature: float = 1.0, seed: int = 42) -> List[str]:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    out = []
    for _ in range(n):
        T = min(length_sampler(), max_len-2)
        x = torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device)
        seq = []
        while len(seq) < T:
            logits = model(x)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            token = idx.item()
            if token == stoi[EOS]:
                break
            if token in (stoi[PAD], stoi[BOS]):
                continue
            seq.append(itos[token])
            x = torch.cat([x, idx], dim=1)
        out.append("".join(seq))
    return out


def build_quality_calibrator(mean_quals: List[float]):
    # Simple bootstrap-style sampler: pick a mean, generate normal per-base around it.
    def sampler(length: int, rng: random.Random):
        mu = rng.choice(mean_quals)
        return [max(5, min(40, int(rng.gauss(mu, 3)))) for _ in range(length)]
    return sampler


def synthesize_fastq_transformer(records: Iterable[Tuple[str, str, List[int]]], num_reads: Optional[int] = None, seed: int = 42,
                                 cfg: Optional[FASTQTransformerConfig] = None):
    if cfg is None:
        cfg = FASTQTransformerConfig()
    rng = random.Random(seed)
    seqs, lens, means = [], [], []
    for _, s, q in records:
        s = s.strip().upper()
        if not s:
            continue
        seqs.append(s)
        lens.append(len(s))
        means.append(sum(q)/len(q))
    if not seqs:
        return []

    # Train transformer LM over DNA alphabet
    model = train_transformer(seqs, cfg)
    # Length sampler draws from empirical distribution
    def length_sampler():
        return rng.choice(lens)
    # Quality calibrator
    q_sampler = build_quality_calibrator(means)

    n = num_reads or len(seqs)
    fake = []
    gens = sample_sequences(model, n=n, length_sampler=length_sampler, max_len=cfg.max_len, temperature=1.0, seed=seed)
    for i, s in enumerate(gens):
        q = q_sampler(len(s), rng)
        fake.append((f"syn_read_tr_{i}", s, q))
    return fake



# =============================================
# FILE: mmai/config.py (UPDATED)
# =============================================
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class SynthConfig:
    model: str = "copulagan"  # "copulagan" | "ctgan"
    epochs: int = 300
    batch_size: int = 512
    seed: int = 42
    num_rows: Optional[int] = None
    device: str = "auto"  # NEW: "cpu" | "cuda" | "auto"











