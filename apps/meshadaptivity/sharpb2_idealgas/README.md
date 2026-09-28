# Axisymmetric ideal-gas Sharp-B mesh adaptivity

This Text2Code application reproduces
`examples/MeshAdaptivity/sharpb2_idealgas/pdeapp.m`. It uses the same curved
two-block quadrilateral mesh, initial state, eight-rank HDG solve, nine-step
artificial-viscosity continuation, and eight backend mesh movements.

Generate the binary inputs and model library, then run from this directory:

```sh
export EXASIM_PREFIX=/path/to/exasim_install
$EXASIM_PREFIX/bin/text2code pdeapp.txt
mpirun -np 8 $EXASIM_PREFIX/bin/exasimapp pdeapp.txt
```

The final adapted result is written to `dataout/outvis.pvtu`. ParaView uses
the following field mapping:

- `Scalar Field 0`: physical density `[kg/m^3]`
- `Scalar Field 1`: physical pressure `[Pa]`
- `Scalar Field 2`: physical temperature `[K]`
- `Scalar Field 3`: Mach number
- `Scalar Field 4`: artificial viscosity
- `Vector Field 0`: physical velocity `[m/s]`

Compare the final distributed fields against the MATLAB reference run:

```sh
python3 verify_against_matlab.py
```

The included `partition.bin` reproduces the MATLAB rank decomposition for a
controlled numerical comparison. The verifier also assembles owned elements
and matches them by global element ID before computing errors.
