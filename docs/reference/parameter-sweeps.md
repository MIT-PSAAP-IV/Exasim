# Parameter Sweep Reference

Exasim supports parameter sweeps over `physicsparam` in two ways:

1. Frontend-driven sweeps in MATLAB/Python/Julia.
2. Standalone C++ sweeps driven by `physicsparamcases.bin`.

Both workflows run one simulation per concrete `physicsparam` vector.

## Frontend Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `physicsparam` | vector | frontend default | Base parameter vector. |
| `physicsparamsweep` | language-specific spec | empty | Structured sweep or list of cases. |
| `physicsparamwarmstart` | int | `0` | Reuse previous case solution when supported. |

Frontend sweeps loop in the frontend, update `physicsparam`, write case output
directories, run the executable, and fetch each case solution.

## Text2Code / Standalone Fields

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `physicsparam` | list(float) | required | Base vector and expected case width. |
| `physicsparamcases` | matrix(float) | empty | One row per sweep case. |
| `physicsparamwarmstart` | int | `0` | Warm-start later cases from previous converged solution. |

## `physicsparamcases` Format

Accepted forms:

```text
physicsparamcases = [[1.0, 2.0], [3.0, 4.0]];
```

or:

```text
physicsparamcases = [1.0, 2.0; 3.0, 4.0];
```

Every row must have the same number of entries as `physicsparam`.

## Runtime Binary Format

`physicsparamcases.bin` stores:

1. Number of cases.
2. Number of parameters.
3. All parameter values in case-major order.

The standalone executable detects this file and runs the sweep internally.

## Output Layout

```text
dataout/
├── physicsparam_sweep_manifest.txt
├── paramcase_0001/
│   ├── physicsparam.txt
│   └── out...
├── paramcase_0002/
│   ├── physicsparam.txt
│   └── out...
└── paramcase_0003/
    ├── physicsparam.txt
    └── out...
```

## Warm-Start Semantics

When `physicsparamwarmstart = 0`:

- Each case uses the normal initialization procedure.
- Initial-condition functions can depend on that case's `physicsparam`.

When `physicsparamwarmstart = 1`:

- The first case uses the normal initialization procedure.
- Later cases reuse the previous case's converged solution as the initial
  state.
- The active `physicsparam` is updated for the next solve.

Warm-start is most useful for continuation studies where adjacent parameter
points have similar solutions.

## GPU Correctness Requirement

When a sweep updates `physicsparam`, both host and device copies must be
refreshed before kernels run. This is a runtime requirement, not a frontend
detail.

## Related Pages

- [pdeapp reference](pdeapp.md)
- [Application-mode parameter sweeps](../usage-modes/parameter-sweeps.md)
- [Runtime internals](../internals/runtime.md)
