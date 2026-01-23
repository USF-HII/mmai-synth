from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class SynthConfig:
    model: str = "copulagan"  # "copulagan" | "ctgan"
    epochs: int = 300
    batch_size: int = 512
    seed: int = 42
    num_rows: Optional[int] = None  # rows to synthesize; if None, match input
