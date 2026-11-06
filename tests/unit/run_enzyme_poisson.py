"""Run a staged Poisson case and check convergence and its solution error."""
import math
from pathlib import Path
import re
import subprocess
import sys

executable, directory = sys.argv[1:]
directory = Path(directory)
result = subprocess.run([executable, "pdeapp.txt"], cwd=directory,
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(directory / "run.log").write_text(result.stdout)
print(result.stdout)
if result.returncode:
    raise SystemExit(result.returncode)
norms = re.findall(r"Residual Norm:\s*([\d.eE+\-]+)", result.stdout)
if not norms or not math.isfinite(float(norms[-1])) or float(norms[-1]) >= 1e-6:
    raise SystemExit("Poisson residual did not meet NewtonTol=1e-6")
outputs = list(directory.rglob("outqoi.txt"))
if len(outputs) != 1:
    raise SystemExit(f"Expected one QoI output, found {outputs}")
last = outputs[0].read_text().strip().splitlines()[-1].split()
error_squared = float(last[1])
if not math.isfinite(error_squared) or not 0 <= error_squared < 1e-8:
    raise SystemExit(f"Poisson L2 error squared is too large: {error_squared}")
print(f"PASS: Enzyme LDG solve residual={norms[-1]}, L2 error squared={error_squared:.6e}")
