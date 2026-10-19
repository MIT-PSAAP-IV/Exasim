# Reacting ISOQ 2D

This Text2Code application defines a two-dimensional finite-rate reacting-flow
model for the ISOQ geometry. The custom `pdemodel.txt` evolves eight conserved
components and one auxiliary thermodynamic variable using HDG at polynomial
order 2. `pdeapp.txt` uses one rank and reads the supplied mesh, conservative
state, artificial-viscosity field, and temperature field from `grid.bin`,
`udg.bin`, `vdg.bin`, and `wdg.bin`.

Run it with an installed Text2Code toolchain:

```sh
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 1 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

This is a generated chemistry model with a large source definition; it is not
equivalent to a built-in ideal-gas Navier-Stokes case.
