# Coupled heat transfer: Isoq2D_matching

Text2Code version of `examples/CoupledProblems/HeatTransfer/Isoq2D_matching`, exported from
the MATLAB example with `exporttext2code`. Axisymmetric Navier-Stokes flow over the ISOQ body (quadrilaterals, model 1) coupled to steady heat conduction in the solid (Gmsh triangles, model 2) through a geometrically conformal interface (`interfacetype = 1`).

As in the MATLAB `pdeapp.m`, model 1 is first solved alone:

- **First stage** (`pdeapp0.txt`): the flow solve with artificial-viscosity
  continuation (`AV = 2`, 8 continuation steps, wall distance from
  `AVdistfunction = 1`, Helmholtz smoothing), starting from `udg0.bin` and
  `vdg0.bin`. Its state and external fields (wall distance, filtered AV
  sensor, mesh-adaptation pressure) become `udg1.bin` and `vdg1.bin`.
- **Coupled solve** (`pdeapp1.txt`, `pdeapp2.txt`): model 1 starts from those
  files with the AV field frozen (`AV = 0`, no distance or smoothing solves,
  which are not safe inside a two-model run).

Run the two stages as two commands, like the two `exasim` calls in the MATLAB
`pdeapp.m`. The first command solves `pdeapp0.txt` alone (data in `datain/` and
`dataout/`) and gathers its result (the last `paramcase_NNNN/` of a sweep) in
global element order into `udg1.bin` and, when the stage writes external fields,
`vdg1.bin`. The second command runs the coupled solve; `pdeapp1.txt` reads those
files as `udgfile`/`vdgfile`, and Text2Code repartitions them, so the two commands
may use different rank counts for model 1.

```sh
cmake -S . -B build -DExasim_DIR=/path/to/exasim-prefix/lib/cmake/Exasim
cmake --build build
mpirun -np 4 build/exasimapp pdeapp0.txt 4
mpirun -np 6 build/exasimapp pdeapp1.txt 4 pdeapp2.txt 2
```

The coupled solutions are written to `dataout1/` (flow) and `dataout2/` (solid).

The solid boundaries are listed as bottom, right, interface: Text2Code assigns
each boundary face to the first expression it satisfies, so the interface is
the last, catch-all expression `x>-1e9`.
