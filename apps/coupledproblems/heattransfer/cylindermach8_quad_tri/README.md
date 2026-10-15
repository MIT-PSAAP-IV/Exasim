# Coupled heat transfer: cylindermach8_quad_tri

Text2Code version of `examples/CoupledProblems/HeatTransfer/cylindermach8_quad_tri`, exported from
the MATLAB example with `exporttext2code`. Mach-8 flow over a cylinder (quadrilaterals, model 1) coupled to heat conduction in the cylinder wall (triangles, model 2) through a non-matching interface (`interfacetype = 1`).

The flow model uses the artificial viscosity `av = c tanh(k d)`, with the wall
distance `d` in `v(1)` (`vdg0.bin`) and `c`, `k` the last two `physicsparam`
entries. As in the MATLAB `pdeapp1.m`, the flow is first converged alone by an
artificial-viscosity continuation:

- **First stage** (`pdeapp0.txt`): a physicsparam sweep over
  `[c k] = [0.06 30; 0.04 30; 0.025 30; 0.025 5]` (`physicsparamcases`), each
  case warm-started from the previous one (`physicsparamwarmstart = 1`),
  starting from `udg0.bin`. One Text2Code run and one solver run.
- **Coupled solve** (`pdeapp1.txt`, `pdeapp2.txt`): model 1 starts from the last
  case (`udg1.bin`) with `[c k] = [0.025 5]`.

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
mpirun -np 2 build/exasimapp pdeapp0.txt 2
mpirun -np 4 build/exasimapp pdeapp1.txt 2 pdeapp2.txt 2
```

The coupled solutions are written to `dataout1/` (flow) and `dataout2/` (solid).
Text2Code writes its generated code into the shared install directory, so run
one Text2Code app at a time.
