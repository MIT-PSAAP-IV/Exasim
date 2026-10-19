# Sharp-body hypersonic flow

This Text2Code case solves a steady hypersonic sharp-body configuration with
built-in model `11`. It uses HDG at polynomial order 2, carries two external
fields, and is configured for four MPI ranks. The free-stream Mach number in
`physicsparam` is approximately 21.38. The mesh and initial fields are supplied
as binary files in this directory.

Run with the installed executables:

```sh
export EXASIM_PREFIX=/path/to/exasim-prefix
"$EXASIM_PREFIX/bin/text2code" pdeapp.txt
mpiexec -n 4 "$EXASIM_PREFIX/bin/exasimapp" pdeapp.txt
```

Alternatively, build the case-local `main.cpp`:

```sh
cmake -S . -B build -DExasim_DIR="$EXASIM_PREFIX/lib/cmake/Exasim"
cmake --build build
mpiexec -n 4 build/exasimapp pdeapp.txt
```

The initial and external fields are part of the problem definition and should
be retained when comparing solver behavior.
