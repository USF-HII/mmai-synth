from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "example_data"
OUTPUT_DIR = ROOT / "synthetic_data"

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
