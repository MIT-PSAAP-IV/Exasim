# Compatibility shim for legacy pdeapps that exec() this file after computing
# cdir/ii. The frontend now lives in the `exasim` package
# (frontends/Python/exasim); new code should simply `import exasim`.
import sys, os

_srcdir = cdir[0:(ii+6)] + "/frontends/Python"
if _srcdir not in sys.path:
    sys.path.insert(0, _srcdir)
sys.path.append(cdir)

import exasim
# Re-export the historical bare module names for un-migrated pdeapps.
Preprocessing = exasim.Preprocessing
Postprocessing = exasim.Postprocessing
Gencode = exasim.Gencode
Mesh = exasim.Mesh
sys.modules.setdefault('Preprocessing', exasim.Preprocessing)
sys.modules.setdefault('Postprocessing', exasim.Postprocessing)
sys.modules.setdefault('Gencode', exasim.Gencode)
sys.modules.setdefault('Mesh', exasim.Mesh)

print('==> Exasim Python frontend (exasim package) ...\n')
