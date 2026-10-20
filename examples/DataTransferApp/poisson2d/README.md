# Poisson data-transfer application

This case is the two-dimensional HDG Poisson example configured to export a
self-contained data-transfer bundle. The bundle contains generated kernels,
input data, a Text2Code model, CMake build files, a manifest, and `run.sh`, so it
can be moved to another machine with an Exasim installation.

MATLAB, Python, and Julia application/model pairs are provided. See
[`../README.md`](../README.md) for the bundle format and relocation workflow.

## Run

```bash
export EXASIM_ROOT=/path/to/Exasim
cd "$EXASIM_ROOT/examples/DataTransferApp/poisson2d"
matlab -batch "pdeapp"
PYTHONPATH="$EXASIM_ROOT/frontends/Python" python3 pdeapp.py
julia --project="$EXASIM_ROOT/frontends/Julia/Exasim" pdeapp.jl
```

Each frontend writes a `poisson2d-bundle`. Run them separately or rename each
bundle before starting the next frontend. Build and execute an exported bundle
with `EXASIM_ROOT=/path/to/install ./run.sh`.
