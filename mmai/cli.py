# =============================================
# FILE: mmai/cli.py (UPDATED: transformer FASTQ, privacy/utility, schema, GPU)
# =============================================
from pathlib import Path
from typing import Optional, List
import typer
import pandas as pd

from .utils.logging import get_logger, console
from .utils.paths import ensure_dirs, OUTPUT_DIR
from .config import SynthConfig
from .io.loaders import detect_type, load_csv, load_plink, fastq_to_dataframe, iter_fastq
from .io.writers import write_csv, write_fastq
from .synth.tabular import synthesize_copulagan, synthesize_ctgan, synthesize_plink
from .synth.genomics import synthesize_fastq_baseline
from .synth.fastq_transformer import FASTQTransformerConfig, synthesize_fastq_transformer
from .eval.metrics import tabular_report
from .eval.privacy import nearest_neighbor_stats, attribute_disclosure_risk
from .eval.utility import train_on_synth_test_on_real
from .schema.hints import load_hints, apply_hints

app = typer.Typer(help="MMAI synthetic data pipeline")
log = get_logger()

@app.command("synth")
def synth(
    input_path: Path = typer.Argument(..., help="Input file (.csv, .bed, .fastq)"),
    output_dir: Path = typer.Option(OUTPUT_DIR, "--outdir", help="Output directory"),
    model: str = typer.Option("copulagan", "--model", help="Model: copulagan|ctgan|fastq_markov|fastq_transformer"),
    rows: Optional[int] = typer.Option(None, help="Rows/reads to synthesize"),
    epochs: int = typer.Option(300, help="Training epochs (tabular or transformer)"),
    batch_size: int = typer.Option(512, help="Batch size (tabular or transformer)"),
    seed: int = typer.Option(42, help="Random seed"),
    schema_path: Optional[Path] = typer.Option(None, "--schema", help="YAML schema hints for tabular"),
    device: str = typer.Option("auto", "--device", help="cpu|cuda|auto (tabular + transformer)"),
):
    """Synthesize data from INPUT_PATH and write results under OUTPUT_DIR."""
    ensure_dirs()
    ftype = detect_type(input_path)
    cfg = SynthConfig(model=model, epochs=epochs, batch_size=batch_size, seed=seed, num_rows=rows, device=device)
    hints = load_hints(schema_path)
    console().rule(f"[bold]Synthesizing[/bold] {input_path.name} ({ftype})")
    output_dir.mkdir(parents=True, exist_ok=True)

    if ftype == "csv":
        df = load_csv(input_path)
        model_df, passthrough_df, discrete_cols = apply_hints(df, input_path.name, hints)
        log.info(f"Loaded CSV with shape {df.shape}; model_df={model_df.shape}; categorical={len(discrete_cols)}")
        if cfg.model.lower() == "ctgan":
            fake = synthesize_ctgan(model_df, epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed, num_rows=cfg.num_rows, discrete_override=discrete_cols, device=cfg.device)
        else:
            fake = synthesize_copulagan(model_df, epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed, num_rows=cfg.num_rows, discrete_override=discrete_cols, device=cfg.device)
        out_csv = output_dir / f"{input_path.stem}.synthetic.csv"
        write_csv(fake, out_csv)
        log.info(f"Wrote: {out_csv}")
        report = tabular_report(model_df, fake)
        log.info(f"Eval: pairwise_corr_delta_mean={report.get('pairwise_corr_delta_mean')}")

    elif ftype == "plink":
        geno_df, meta = load_plink(input_path)
        log.info(f"Loaded PLINK: samples x snps = {geno_df.shape}")
        fake = synthesize_plink(geno_df, epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed, num_rows=cfg.num_rows, device=cfg.device)
        out_csv = output_dir / f"{input_path.stem}.synthetic.plink.csv"
        write_csv(fake, out_csv)
        log.info(f"Wrote genotype table: {out_csv}")
        report = tabular_report(geno_df, fake)
        log.info(f"Eval: pairwise_corr_delta_mean={report.get('pairwise_corr_delta_mean')}")

    elif ftype == "fastq":
        reads = list(iter_fastq(input_path))
        if model.lower() in ("fastq_transformer", "transformer"):
            tr_cfg = FASTQTransformerConfig(epochs=epochs, batch_size=batch_size, device=device)
            fake_reads = synthesize_fastq_transformer(reads, num_reads=rows, seed=seed, cfg=tr_cfg)
        else:
            fake_reads = synthesize_fastq_baseline(reads, num_reads=rows, seed=seed)
        out_fq = output_dir / f"{input_path.stem}.synthetic.fastq"
        write_fastq(fake_reads, out_fq)
        log.info(f"Wrote FASTQ: {out_fq}")
    else:
        raise typer.BadParameter(f"Unsupported type: {ftype}")


@app.command("privacy")
def privacy(
    real_path: Path = typer.Argument(..., help="Real CSV or PLINK (.bed)"),
    fake_path: Path = typer.Argument(..., help="Synthetic CSV (from this tool)"),
):
    """Nearest-neighbor privacy stats & attribute disclosure proxy."""
    if real_path.suffix == ".bed":
        from .io.loaders import load_plink
        real_df, _ = load_plink(real_path)
    else:
        real_df = pd.read_csv(real_path)
    fake_df = pd.read_csv(fake_path)
    nn_stats = nearest_neighbor_stats(real_df, fake_df)
    log.info(f"NN privacy: {nn_stats}")
    console().print(nn_stats)


@app.command("utility")
def utility(
    real_path: Path = typer.Argument(..., help="Real CSV or PLINK (.bed)"),
    fake_path: Path = typer.Argument(..., help="Synthetic CSV (from this tool)"),
    target: str = typer.Argument(..., help="Target column for classification utility benchmark"),
):
    """Train on synthetic, test on real (LogReg baseline, numeric-only)."""
    if real_path.suffix == ".bed":
        from .io.loaders import load_plink
        real_df, _ = load_plink(real_path)
    else:
        real_df = pd.read_csv(real_path)
    fake_df = pd.read_csv(fake_path)
    res = train_on_synth_test_on_real(real_df, fake_df, target)
    console().print(res)