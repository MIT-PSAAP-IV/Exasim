# Cylinder Mach 8 mesh adaptivity (text2code)

This text2code application is exported from
`examples/MeshAdaptivity/cylindermach8/pdeapp_backend.m`. It runs the same
ten-step artificial-viscosity continuation and performs one backend mesh
movement after each of the first nine flow solves.

From this directory, generate the binary input and model library, then run:

```sh
export EXASIM_PREFIX=/path/to/exasim_install
$EXASIM_PREFIX/bin/text2code pdeapp.txt
$EXASIM_PREFIX/bin/exasimapp pdeapp.txt
```

To compare the completed text2code run with a completed MATLAB backend run:

```sh
python3 verify_against_matlab.py
```

The verifier compares the final adapted geometry, flow solution, and external
AV fields against `examples/MeshAdaptivity/cylindermach8/backend_run/dataout`.
It fails if any relative L2 error exceeds `1e-6`.

The high-level inputs can be regenerated from MATLAB without running the
solver:

```matlab
text2code_export_directory = fullfile(EXASIM_ROOT,'apps','meshadaptivity','cylindermach8');
text2code_export_only = true;
run(fullfile(EXASIM_ROOT,'examples','MeshAdaptivity','cylindermach8','pdeapp_backend.m'));
```
