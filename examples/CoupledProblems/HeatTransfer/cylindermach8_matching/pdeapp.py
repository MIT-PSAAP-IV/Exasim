"""Python coupled Mach-8 cylinder heat-transfer example (matching meshes)."""
import os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); REPO=os.path.abspath(os.path.join(ROOT,"..","..",".."))
sys.path[:0]=[ROOT,os.path.join(REPO,"frontends","Python")]
import exasim
from cylinder_driver import run_case
def main(): return run_case("matching",HERE)
if __name__ == "__main__": main()
