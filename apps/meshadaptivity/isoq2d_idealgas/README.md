# Axisymmetric ideal-gas ISOQ mesh adaptivity

This Text2Code application reproduces
`examples/MeshAdaptivity/isoq2d_idealgas/pdeapp.m`. It uses the same curved
two-block quadrilateral mesh, initial state, four-rank HDG solve, six-step
artificial-viscosity continuation, and five backend mesh movements.

Generate the binary inputs and model library, then run from this directory:

```sh
export EXASIM_PREFIX=/path/to/exasim_install
$EXASIM_PREFIX/bin/text2code pdeapp.txt
mpirun -np 4 $EXASIM_PREFIX/bin/exasimapp pdeapp.txt
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

The verifier assembles owned elements across ranks and matches elements by
global element ID, so it does not require identical METIS partitions. Because
the HDG nonlinear/iterative path is not bitwise invariant to the MPI
decomposition, independently generated MATLAB and Text2Code partitions can
produce larger final differences than a controlled same-partition comparison.
