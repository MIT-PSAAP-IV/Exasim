"""Run the canonical staged ideal-gas-cylinder workflow in isolation."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _case_launcher import run_matlab

def main():
    return run_matlab(Path(__file__), HERE, "pdeapp.m", HERE.parent / "idealgascylinder_python_run")

if __name__ == "__main__":
    main()
