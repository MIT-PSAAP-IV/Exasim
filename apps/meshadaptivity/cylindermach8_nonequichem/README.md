# Finite-rate five-species-air cylinder Mach 8 mesh adaptivity

This Text2Code application reproduces the backend shock-capturing and mesh-
adaptivity workflow configured by
`examples/MeshAdaptivity/cylindermach8_nonequichem/pdeapp.m`. It uses the
same second-order quadrilateral mesh, initial state, eight-rank HDG solve,
ten-step artificial-viscosity continuation, and nine mesh movements.

Generate the binary inputs and model library, then run from this directory:

```sh
export EXASIM_PREFIX=/path/to/exasim_install
$EXASIM_PREFIX/bin/text2code pdeapp.txt
mpirun -np 8 $EXASIM_PREFIX/bin/exasimapp pdeapp.txt
```

The final adapted result is written to `dataout/outvis_000003.pvtu`. ParaView uses
the following generated field mapping:

- `Scalar Field 0`: physical mixture density `[kg/m^3]`
- `Scalar Field 1`: physical pressure `[Pa]`
- `Scalar Field 2`: physical temperature `[K]`
- `Scalar Field 3`: frozen-composition Mach number
- `Scalar Field 4`: artificial viscosity
- `Scalar Field 5`: atomic-nitrogen mass fraction `Y_N`
- `Scalar Field 6`: atomic-oxygen mass fraction `Y_O`
- `Scalar Field 7`: nitric-oxide mass fraction `Y_NO`
- `Scalar Field 8`: molecular-nitrogen mass fraction `Y_N2`
- `Scalar Field 9`: molecular-oxygen mass fraction `Y_O2`
- `Vector Field 0`: physical velocity `[m/s]`

Compare the final distributed fields against the MATLAB reference run:

```sh
python3 verify_against_matlab.py
```

The comparison uses the global element IDs stored in each mesh partition, so
it does not require MATLAB and Text2Code to produce identical METIS partitions.
