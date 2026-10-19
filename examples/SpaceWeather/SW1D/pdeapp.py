"""Run the canonical SW1D space-weather workflow in isolation."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; REPOSITORY=HERE.parents[2]
sys.path.insert(0,str(REPOSITORY/"examples"/"NavierStokes"))
from _case_launcher import run_matlab
def main(): return run_matlab(Path(__file__),HERE,"pdeapp.m",HERE.parent/"SW1D_python_run")
if __name__=="__main__": main()
