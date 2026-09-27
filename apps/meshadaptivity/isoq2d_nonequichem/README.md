# Axisymmetric finite-rate-air ISOQ mesh adaptivity

This Text2Code application reproduces
`examples/MeshAdaptivity/isoq2d_nonequichem/pdeapp.m`. It uses the same
second-order two-block quadrilateral mesh, initial finite-rate five-species-air
state, eight-rank HDG solve, ten-step artificial-viscosity continuation, and
nine backend mesh movements.

`partition.bin` contains the one-based owner rank of every coarse element from
the MATLAB preprocessing run. Text2Code therefore uses the same domain
decomposition as the MATLAB reference instead of invoking an independent METIS
partition.

Generate the model library and binary inputs, then run from this directory:

```sh
export EXASIM_PREFIX=/path/to/exasim_install
$EXASIM_PREFIX/bin/text2code pdeapp.txt
EXASIM_MESHADAPT_VERIFY=1 mpirun -np 8 \
  $EXASIM_PREFIX/bin/exasimapp pdeapp.txt
```

The final adapted result is written under `dataout`. ParaView uses this field
mapping:

- `Scalar Field 0`: physical pressure `[Pa]`
- `Scalar Field 1`: physical mixture density `[kg/m^3]`
- `Scalar Field 2`: physical temperature `[K]`
- `Scalar Field 3`: frozen-composition Mach number
- `Scalar Fields 4-8`: `Y_N`, `Y_O`, `Y_NO`, `Y_N2`, and `Y_O2`
- `Scalar Field 9`: artificial viscosity
- `Vector Field 0`: physical velocity `[m/s]`

Compare the Text2Code result and partition against the existing MATLAB run:

```sh
python3 verify_against_matlab.py
```

The verifier reorders distributed fields by global element ID and reports
relative L2 and maximum absolute differences without enforcing a tolerance.
