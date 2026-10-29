"""Native Python model generation for the five clean-start examples.

This generates BOTH native kernel families. The runtime input independently
selects pure HDG, pure LDG or the two rank-separated regions. Numerical model
parameters stay in the common runtime configuration; only their symbolic
count is needed here. No Text2Code exporter or compiler is called.
"""

from pathlib import Path
import argparse
import importlib.util
import sys

import numpy as np
import exasim


def generate_python(case_dir, build_dir, nd, ncu, nparam):
    case = Path(case_dir).resolve(strict=True)
    output = Path(build_dir)
    if (not output.is_absolute() or output.exists() or
            not output.resolve().is_relative_to(Path("/tmp").resolve())):
        raise ValueError("model build directory must be new and absolute under /tmp")
    if nd not in (2, 3) or ncu < 1 or nparam < 0:
        raise ValueError("invalid ModelD dimensions")
    model_file = case / "pdemodel.py"
    if not model_file.is_file():
        raise ValueError("missing native Python model: " + str(model_file))

    pde, _ = exasim.initializeexasim()
    pde.update(model="ModelD", hybrid=1, modelnumber=0, nd=nd, ncx=nd,
               ncu=ncu, nch=ncu, ncq=nd*ncu, nc=(nd+1)*ncu,
               nco=0, ncw=0, nce=0, nsca=0, nvec=0, nten=0,
               nbqoi=0, nvqoi=0, nsurfq=0, nmaterialstate=0,
               physicsparam=np.zeros(nparam), uinf=np.zeros(0), tau=np.ones(1),
               builddir=str(output), buildpath=str(output), datapath=str(output))
    # A case-specific import name avoids silently reusing another case's
    # cached pdemodel when a Python caller generates more than one model.
    module_name = "_hdgldg_" + case.name
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    sys.path.insert(0, str(case))
    try:
        spec.loader.exec_module(module)
        pde["modelfile"] = module_name
        exasim.Gencode.gencode(pde)
    finally:
        sys.path.pop(0)
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return str(output / "kernels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("build_dir")
    parser.add_argument("nd", type=int)
    parser.add_argument("ncu", type=int)
    parser.add_argument("nparam", type=int)
    arguments = parser.parse_args()
    print(generate_python(arguments.case_dir, arguments.build_dir,
                          arguments.nd, arguments.ncu, arguments.nparam))
