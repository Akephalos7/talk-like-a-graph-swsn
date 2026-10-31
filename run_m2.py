from pathlib import Path
from swsn_m2.pipeline import run_pipeline

if __name__ == '__main__':
    run_pipeline(Path(__file__).parent)
