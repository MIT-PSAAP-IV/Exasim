# Equilibrium-air cylinder Mach 8 mesh adaptivity (Text2Code)

This application is exported from
`examples/MeshAdaptivity/cylindermach8_equichem/pdeapp.m`. It uses the same
mesh, equilibrium-air material database, initial condition, four-rank HDG
configuration, ten-step artificial-viscosity continuation, and nine backend
mesh movements as the MATLAB example.

The final adapted solution is written to `dataout/outvis.pvtu`. ParaView uses
the following field mapping:

- `Scalar Field 0`: physical density `[kg/m^3]`
- `Scalar Field 1`: physical pressure `[Pa]`
- `Scalar Field 2`: physical temperature `[K]`
- `Scalar Field 3`: Mach number
- `Scalar Field 4`: atomic-nitrogen mass fraction `Y_N`
- `Scalar Field 5`: atomic-oxygen mass fraction `Y_O`
- `Scalar Field 6`: nitric-oxide mass fraction `Y_NO`
- `Scalar Field 7`: molecular-nitrogen mass fraction `Y_N2`
- `Scalar Field 8`: molecular-oxygen mass fraction `Y_O2`
- `Vector Field 0`: physical velocity `[m/s]`

From this directory, generate the binary inputs and model library, then run:

```sh
export EXASIM_PREFIX=/path/to/exasim_install
$EXASIM_PREFIX/bin/text2code pdeapp.txt
EXASIM_MESHADAPT_VERIFY=1 mpirun -np 4 \
  $EXASIM_PREFIX/bin/exasimapp pdeapp.txt
```

Compare the completed run with the MATLAB reference output:

```sh
python3 verify_against_matlab.py
```

The MATLAB and Text2Code preprocessors may produce different valid METIS
partitions. The verifier therefore assembles owned elements across ranks,
matches elements by physical coordinates, and reports relative L2 and maximum
absolute differences without enforcing a pass/fail tolerance.

Regenerate the high-level Text2Code inputs without running the MATLAB solver:

```matlab
text2code_export_directory = fullfile(EXASIM_ROOT, 'apps', ...
    'meshadaptivity', 'cylindermach8_equichem');
text2code_export_only = true;
run(fullfile(EXASIM_ROOT, 'examples', 'MeshAdaptivity', ...
    'cylindermach8_equichem', 'pdeapp.m'));
```
