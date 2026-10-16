# Coupled heat transfer: cylindermach8_nonmatching

Text2Code version of `examples/CoupledProblems/HeatTransfer/cylindermach8_nonmatching`,
exported from the MATLAB example with `exporttext2code`. Mach-8 flow over a
cylinder (868 quadrilaterals, 28 interface faces, model 1) is coupled to heat
conduction in the cylinder wall (105 quadrilaterals, 21 interface faces, model 2)
through a non-matching interface (`interfacetype = 1`). Both interfaces lie on the
same radius-one semicircle, giving a 4:3 face ratio without a gap or overlap; the
solver locates donor points on the other model's mesh and prints the
"Non-matching donor search, Gamma^1" and "Gamma^2" diagnostics during setup. Both
models use polynomial order 4 and the same face quadrature.

As in the MATLAB `pdeapp1.m`, the flow (model 1) is first converged alone by an
artificial-viscosity continuation. The flow model uses `av = c tanh(k d)`, with
the wall distance `d` in `v(1)` (`vdg0.bin`) and `c`, `k` the last two
`physicsparam` entries.

- **First stage** (`pdeapp0.txt`): a physicsparam sweep over
  `[c k] = [0.06 30; 0.04 30; 0.025 30; 0.025 5]` (`physicsparamcases`), each
  case warm-started from the previous one (`physicsparamwarmstart = 1`),
  starting from `udg0.bin`. Its final state (the last case) becomes `udg1.bin`.
- **Coupled solve** (`pdeapp1.txt`, `pdeapp2.txt`): model 1 starts from
  `udg1.bin` with `[c k] = [0.025 5]` and the same wall distance (`vdg0.bin`);
  model 2 starts from the uniform wall temperature in `udg2.bin`.

Run the two stages as two commands, like the two `exasim` calls in the MATLAB
example. The first command solves `pdeapp0.txt` alone (data in `datain/` and
`dataout/`); `main.cpp` then gathers the last sweep case (`paramcase_0004/`) in
global element order into `udg1.bin`. The second command runs the coupled solve;
`pdeapp1.txt` reads `udg1.bin` as `udgfile`, and Text2Code repartitions it, so the
two commands may use different rank counts for model 1.

```sh
cmake -S . -B build -DExasim_DIR=/path/to/exasim-prefix/lib/cmake/Exasim
cmake --build build
mpirun -np 2 build/exasimapp pdeapp0.txt 2
mpirun -np 4 build/exasimapp pdeapp1.txt 2 pdeapp2.txt 2
```

The coupled solutions are written to `dataout1/` (flow) and `dataout2/` (solid).
