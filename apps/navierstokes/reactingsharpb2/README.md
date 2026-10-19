# Reacting sharp-body flow

This application uses built-in model `4` for a five-species reacting-flow
calculation over the sharp-body geometry. It uses HDG, polynomial order 2,
quadrature order 4, pseudo-time continuation, and four MPI ranks. The supplied
`grid.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`, and `xdg.bin` are part of the
canonical starting state.

```sh
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 4 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Run from an isolated copy if preserving existing local output is important;
Text2Code and the solver create `datain/` and `dataout/` in the case directory.
