"""Generate a real Python frontend kernel set without running a PDE solve."""
import argparse
from pathlib import Path
import numpy as np
from exasim.Preprocessing.initializepde import initializepde
from exasim.Gencode.Gencode import gencode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
pde = initializepde("enzyme-test")
pde.update(builddir=str(args.output.resolve()), modelfile="python_model",
           nd=2, ncx=2, ncu=2, nc=6, ncq=4, nch=2, nco=1, ncw=1,
           nce=1, physicsparam=np.array([2.0]), uinf=np.array([1.0]),
           tau=np.array([1.5]), modelnumber=0, hybrid=1)
gencode(pde)
print(args.output.resolve() / "kernels")
