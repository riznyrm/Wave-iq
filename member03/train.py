"""Train the Member 03 models: python train.py"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT / "src"))
from waveiq.training import run_training
from waveiq.plots import create_plots
from waveiq.reporting import write_results

if __name__ == "__main__":
    run_training(ROOT)
    create_plots(ROOT)
    write_results(ROOT)
