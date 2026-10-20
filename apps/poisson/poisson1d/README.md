# One-dimensional Poisson problem

This is a minimal custom Text2Code Poisson application on the one-dimensional
mesh stored in `grid.bin`. It uses HDG, polynomial order 2, quadrature order 4,
and one rank. The source term and exact Dirichlet trace are defined in
`pdemodel.txt`.

The directory does not contain a standalone CMake project. Run it with an
installed toolchain:

```sh
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 1 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

This small problem is suitable for checking parsing and basic CPU execution,
but this README does not claim a recorded numerical tolerance.
