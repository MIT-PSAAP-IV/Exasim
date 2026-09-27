# Axisymmetric equilibrium-air ISOQ mesh adaptivity

This Text2Code application reproduces
`examples/MeshAdaptivity/isoq2d_equichem/pdeapp.m`. It uses the same curved
two-block quadrilateral mesh, equilibrium five-species-air material database,
initial state, four-rank HDG solve, ten-step artificial-viscosity continuation,
and nine backend mesh movements.

Generate the binary inputs and model library, then run from this directory:

```sh
export EXASIM_PREFIX=/path/to/exasim_install
$EXASIM_PREFIX/bin/text2code pdeapp.txt
EXASIM_MESHADAPT_VERIFY=1 mpirun -np 4 \
  $EXASIM_PREFIX/bin/exasimapp pdeapp.txt
```

The final adapted result is written to `dataout/outvis.pvtu`. ParaView uses
the following field mapping:

- `Scalar Field 0`: physical density `[kg/m^3]`
- `Scalar Field 1`: physical pressure `[Pa]`
- `Scalar Field 2`: physical temperature `[K]`
- `Scalar Field 3`: Mach number
- `Scalar Field 4`: artificial viscosity
- `Scalar Field 5`: atomic-nitrogen mass fraction `Y_N`
- `Scalar Field 6`: atomic-oxygen mass fraction `Y_O`
- `Scalar Field 7`: nitric-oxide mass fraction `Y_NO`
- `Scalar Field 8`: molecular-nitrogen mass fraction `Y_N2`
- `Scalar Field 9`: molecular-oxygen mass fraction `Y_O2`
- `Vector Field 0`: physical velocity `[m/s]`

Compare final distributed fields against the MATLAB reference run:

```sh
python3 verify_against_matlab.py
```

The verifier assembles owned elements across ranks and reorders them by global
element ID, so it does not require identical local element counts.
