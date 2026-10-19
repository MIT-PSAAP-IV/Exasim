# Coupled Poisson: nonmatching3d_hex

This is the Text2Code version of `examples/CoupledProblems/Poisson/nonmatching3d_hex`,
exported from the MATLAB example with `exporttext2code`.

Two Poisson models are coupled through the non-matching interface x = 0.5 of the
unit cube (`interfacetype = 1`):

- model 1: [0, 0.5] x [0, 1]^2, 2 x 4 x 4 hexahedra;
- model 2: [0.5, 1] x [0, 1]^2, 2 x 2 x 2 hexahedra.

Both use p = 2. The exact solution is u = exp(x) sin(pi y) sin(pi z).

The run prints the `Non-matching donor search, Gamma^1` and `Gamma^2`
diagnostics for both coupling directions during setup.

Files:

- `pdeapp1.txt`, `pdeapp2.txt`: application, mesh, solver and coupling settings
  of each model.
- `pdemodel1.txt`, `pdemodel2.txt`: PDE model definitions consumed by Text2Code.
- `grid1.bin`, `grid2.bin`: mesh coordinates and connectivity of each model.

Build against an installed Exasim and run the two-domain coupled solver:

```sh
cmake -S . -B build -DExasim_DIR=/path/to/exasim-prefix/lib/cmake/Exasim
cmake --build build
mpirun -np 2 build/exasimapp pdeapp1.txt 1 pdeapp2.txt 1
```

The first run uses Text2Code to generate `datain1/` and `datain2/`. Each run
writes the solutions `dataout1/outudg_np*.bin` and `dataout2/outudg_np*.bin`.
`tests/nonmatching/phase11_text2code.m` checks that this app reproduces the
MATLAB solution to within 2e-11 relative.
