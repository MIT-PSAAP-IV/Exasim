"""Run the canonical steady NACA0012 Text2Code application."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from _case_launcher import run_text2code

def main():
    return run_text2code(Path(__file__), REPOSITORY / "apps" / "navierstokes" / "naca0012steady", 1, HERE / "python_run")

if __name__ == "__main__":
    main()
