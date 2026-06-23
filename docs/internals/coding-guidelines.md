# Coding Guidelines

These guidelines are implementation-focused and supplement the user-facing
documentation. Correctness and backward compatibility take priority over
performance and refactoring.

## General Rules

- Make minimal, localized changes.
- Preserve existing public interfaces unless the change explicitly requires an
  API update.
- Keep frontend-specific logic out of backend solver loops.
- Keep theory derivations out of implementation pages.
- Update tests and documentation with behavior changes.
- Do not commit local build artifacts or generated files unintentionally.

## Data Layout

Exasim commonly uses flattened, column-major numerical arrays with zero-based
C++ indexing. Before changing an array layout, identify all writers, readers,
GPU copies, MPI communication paths, and postprocessing readers.

## Runtime Structs

Runtime structs in `backend/Common/common.h` are shared across many kernels.
When adding fields:

1. Initialize defaults.
2. Serialize and deserialize if the field comes from frontend/Text2Code input.
3. Copy to device if kernels use it on GPU.
4. Preserve struct consistency in postprocess-only paths.
5. Update documentation and tests.

## GPU-Aware Coding

For GPU-sensitive code:

- Use existing allocation/copy helpers.
- Avoid hidden host-device synchronizations inside inner loops.
- Avoid allocating temporaries repeatedly inside Newton or GMRES loops.
- Keep generated kernels free of host-only dependencies.
- Verify CUDA and HIP compile paths when changing provider or kernel code.

## MPI-Aware Coding

For MPI-sensitive code:

- Distinguish owned elements from ghost elements.
- Avoid double-counting ghosts in QoI and integrals.
- Ensure shared files are written rank-safely.
- Preserve rank-local file naming.
- Test more than one rank count when changing DMD or communication paths.

## Generated-Code Rules

Generated code should be small and mechanical. If a generated file needs a
behavior change, prefer modifying:

- The frontend generator.
- Text2Code generation logic.
- CMake app template.
- Provider template.

Only edit generated output directly when it is a checked-in template or a
curated reference implementation.

## Error Handling

Error messages should include:

- The failing file, path, option, or case index.
- Expected dimensions when validating arrays.
- Backend mode when relevant.
- Rank information for MPI-specific errors.

Avoid silent fallback when a malformed runtime file may produce a physically
wrong simulation.

## Documentation Expectations

Update docs when behavior changes:

| Change | Documentation target |
| --- | --- |
| User-facing option | Frontends, Text2Code/reference, relevant usage page. |
| Runtime architecture | Internals. |
| Numerical method | Theory. |
| PDE formulation concept | Physics Models. |
| Installation/package behavior | Installation or Reference. |

## Code Review Focus

Reviewers should pay special attention to:

- CPU/GPU divergence.
- MPI ownership and file races.
- Backward compatibility for existing examples.
- Public installed headers and CMake targets.
- Generated file source-of-truth.
- Whether tests cover the changed execution path.

## Related Pages

- [Development workflow](development-workflow.md)
- [GPU implementation](gpu-implementation.md)
- [MPI implementation](mpi-implementation.md)
- [Testing](testing.md)
