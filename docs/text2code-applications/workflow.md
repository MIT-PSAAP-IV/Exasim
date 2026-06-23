# Workflow And Troubleshooting

This page describes the practical Text2Code workflow: running the executable,
understanding generated files, and debugging common issues.

## Running Text2Code

```bash
/path/to/exasim-prefix/local/bin/text2code pdeapp.txt
```

`pdeapp.txt` points to the companion `pdemodel.txt` through `modelfile`.
Relative paths are interpreted relative to the working directory or app file
path depending on the parsed path settings.

## Generated Artifacts

Common artifacts include:

| Artifact | Purpose |
| --- | --- |
| `datain/app.bin` | Serialized application and solver configuration. |
| `datain/master.bin` | Reference element and quadrature data. |
| `datain/mesh.bin` or rank-local mesh files | Backend mesh and decomposition data. |
| `datain/sol.bin` or rank-local solution files | Geometry nodes and optional initial fields. |
| `datain/physicsparamcases.bin` | Standalone parameter sweep data when `physicsparamcases` is set. |
| generated `.cpp` / `.hpp` files | Model kernels generated from `pdemodel.txt`. |
| `dataout/` | Runtime output from the solver. |

## Development Loop

1. Start from a working app such as `apps/poisson/poisson2d`.
2. Modify `pdemodel.txt` with the smallest physics change.
3. Keep `pdeapp.txt` dimensions and output counts consistent.
4. Run `/path/to/exasim-prefix/local/bin/text2code pdeapp.txt`.
5. Build the generated app or external model.
6. Run a small CPU serial case.
7. Verify residuals, solution fields, and visualization.
8. Scale to MPI/GPU/HPC only after the serial case is correct.

## Typical Commands

```bash
cd apps/poisson/poisson2d
/path/to/exasim-prefix/local/bin/text2code pdeapp.txt
cmake -S . -B build -DExasim_DIR=/path/to/exasim-prefix
cmake --build build -j
build/exasimapp datain/ dataout/out
```

If using an app under `apps/builtinlibrary` or another wrapper, follow that
wrapper's CMake and runtime command format.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `Flux is not defined` or similar | Required function missing from `outputs` or function block. | Add required output and function to `pdemodel.txt`. |
| Function argument error | Argument not declared in `scalars` or `vectors`. | Add the argument declaration or fix the function signature. |
| Output size mismatch | `output_size(...)` inconsistent with `pdeapp.txt` counts. | Align `nsca`, `nvec`, `ncu`, `nvqoi`, etc. |
| Boundary condition behaves incorrectly | `boundaryconditions` IDs do not match model boundary logic. | Check boundary expressions and boundary functions. |
| No VTK output | `saveParaview = 0` or visualization counts/functions missing. | Enable `saveParaview` and add `Vis*` functions. |
| Sweep fails dimension validation | `physicsparamcases` row length differs from `physicsparam`. | Make every case row the same length. |
| MPI run cannot read files | `mpiprocs` at execution differs from preprocessing. | Regenerate `datain` with the intended `mpiprocs`. |
| GPU build fails | Exasim package lacks requested CUDA/HIP variant. | Rebuild/install Exasim with the matching GPU backend. |

## Validating Outputs

For new models, verify:

- residual history decreases for steady nonlinear solves;
- conserved quantities behave correctly when applicable;
- boundary conditions are active on the intended boundaries;
- VTK fields match expected component counts;
- parameter sweep case metadata matches the intended parameters;
- CPU and GPU results agree within expected tolerances before large runs.

## See Also

- [pdeapp.txt guide](pdeapp.md)
- [pdemodel.txt guide](pdemodel.md)
- [Postprocessing](../usage-modes/postprocessing.md)
- [Testing internals](../internals/testing.md)
