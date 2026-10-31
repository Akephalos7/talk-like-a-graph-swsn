from pathlib import Path
import argparse
from swsn_m2.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 2 dataset analysis")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    run_pipeline(Path(__file__).parent, demo=args.demo, download=not args.no_download)
