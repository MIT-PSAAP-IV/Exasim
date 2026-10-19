"""Run the canonical initialized STG steady workflow in isolation."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _case_launcher import run_matlab

def main():
    return run_matlab(Path(__file__), HERE, "pdeapp.m", HERE.parent / "stg_steadystate2d_python_run")

if __name__ == "__main__":
    main()
