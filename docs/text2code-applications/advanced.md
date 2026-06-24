# Advanced Text2Code Topics

Text2Code can support large and complex Exasim applications when the model file,
application file, and generated backend code remain dimensionally consistent.

## Multi-Physics Models

Represent multiphysics systems by packing all primary unknowns into `uq` and
documenting their ordering.

Example ordering:

```text
uq[0] = density
uq[1] = x-momentum
uq[2] = y-momentum
uq[3] = total energy
uq[4:] = gradients or additional coupled variables
```

Set `ncu`, vector sizes, `Flux`, `Source`, and boundary outputs consistently.

## Parameterized PDEs

Use `mu` (`physicsparam`) for values that should change without editing
`pdemodel.txt`:

```text
Re = mu[1];
Pr = mu[2];
Minf = mu[3];
```

Use `physicsparamcases` for runtime sweeps and keep each row the same length as
`physicsparam`.

## User-Defined Constitutive Laws

Complex material models can be written as helper functions in `pdemodel.txt`
and called from generated outputs:

```text
function Viscosity(x, uq, v, w, eta, mu, t)
  output_size(muvisc) = 1;
  muvisc[0] = mu[0]*pow(uq[0], 0.7);
end
```

Only functions listed in `outputs` emit kernels directly, but helper functions
can be called by emitted functions.

## Coupling And Interface Terms

`Fint` and `Fext` support interior/external interface flux hooks used by
coupling workflows. If using these, also configure the corresponding
application fields such as interface conditions, flux maps, or external flux
flags as required by the solver wrapper.

## Large-Scale HPC Runs

For large cases:

- generate and test `datain` on a small mesh first;
- keep generated model code under version control only when it is intended to
  be source, not transient output;
- use installed Exasim packages rather than ad hoc local build paths;
- ensure `mpiprocs`, partition files, and runtime launch ranks match;
- use cluster scratch for large `dataout` trees;
- verify `EXASIM_PREFIX` and package discovery on compute nodes.

## Extending The Text2Code Language

The DSL parser lives in `text2code/text2code/TextParser.hpp`, and code emission
lives mainly in `CodeGenerator.cpp`. Extending the language requires preserving:

- parse determinism;
- SymEngine expression validity;
- generated C++ compatibility with Exasim's model contract;
- Jacobian/Hessian generation semantics;
- backend ABI and output function names.

For maintainability, prefer adding helper functions to `pdemodel.txt` before
adding new grammar features.

## Limitations And Assumptions

- The DSL is line-oriented and simpler than a full programming language.
- Function arguments must be declared in `scalars` or `vectors`.
- Default Exasim mode requires the six core functions in `outputs`.
- Geometry-changing parameter sweeps require regenerating mesh/preprocessing
  data; `physicsparamcases` assumes the mesh and model dimensions are fixed.
- Runtime GPU behavior depends on the installed Exasim CUDA/HIP package, not
  only on `platform = "gpu"`.

## See Also

- [pdemodel.txt syntax](../reference/pdemodel.md)
- [Driving the solver](../driving-the-solver.md)
- [Parameter sweeps](../usage-modes/parameter-sweeps.md)
- [HPC install guide](../install/hpc.md)
