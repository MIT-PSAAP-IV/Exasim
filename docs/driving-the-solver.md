# Driving the solver (C++ API)

The declarative files — [`pdemodel.txt`](usage-modes/index.md) (the model) and
`pdeapp.txt` (the app) — describe *what* to solve. This page covers the
**imperative** side: the `ExasimSolver` C++ class, which lets your own program
construct the solver, step it, and exchange data at a mesh interface. It is the
contract you build against when embedding Exasim in a larger program.

Everything here is in `<exasim/ExasimSolver.hpp>` (pulled in by
`<exasim/ExasimSolverSetup.hpp>`). The skeleton below is illustrative scaffolding,
not a complete coupled solve.

## Lifecycle

A run is a fixed sequence of calls. For an ordinary, self-contained run you do
not write them out — the `RunExasimSolver` helper in `ExasimSolverSetup.hpp` does
it for you (this is what every [usage-mode](usage-modes/index.md) `main.cpp`
calls):

| Step | Method | Purpose |
|---|---|---|
| 1 | `InitializeEnvironment(argc, argv, comm)` | MPI / Kokkos init |
| 2 | `ParseInputs(argc, argv, abi)` | read `pdeapp.txt` and use the selected model ABI to fill missing model sizes |
| 3 | `SetModelDefinition(...)` | attach the same model provider ABI (done by `ConfigureModelDefinitions`) |
| 4 | `InitializeModels()` | build preprocessing data, allocate solution |
| 5 | `Solve()` | run to completion (time stepping / steady / pseudo-time) |
| 6 | `Finalize()` | flush output, finalize MPI / Kokkos |

```cpp
#include <exasim/ExasimSolverSetup.hpp>
#include <exasim/builtinlibprovider.hpp>

ExasimSolver solver;
return RunExasimSolver(solver, argc, argv, comm);   // steps 1–6
```

To drive the solver yourself, use `InitializeExasimSolver(...)` instead — it runs
steps 1–4 and stops, leaving the solver ready to step:

```cpp
ExasimSolver solver;
int err = InitializeExasimSolver(solver, argc, argv, comm);   // steps 1–4
// ... drive the solver ...
solver.Finalize();                                            // step 6
```

## Stepping

Rather than the all-in-one `Solve()`, advance the solver explicitly:

| Method | Behavior |
|---|---|
| `Solve()` | run the configured problem to completion |
| `RunTimeDependent(start, steps)` | advance `steps` time steps starting at step `start` |
| `RunSteady()` | solve the steady problem |
| `RunPseudoTime()` | pseudo-time continuation to steady state |

Per-model overloads (`RunTimeDependent(modelnumber, ...)`, `Solve(modelnumber)`,
…) drive one model when several are configured.

## Interface exchange

These methods let an outside driver read and write data on a designated mesh
interface (an interface is a boundary, selected by its boundary tag `ibc`):

| Method | Purpose |
|---|---|
| `IntializeMeshInterface(modelnumber, ncuext, ncuint, ibc, comperm, offset, comm)` | declare the interface and the field layout exchanged across it |
| `std::vector<ExasimPoint> getInterfacePoints() const` | the coordinates of the interface points (hand these to the other side so it produces matching data) |
| `void getInterfaceFluxes(std::vector<dstype>& send) const` | read out the interface fluxes Exasim computed (to send to the other side) |
| `void setInterfaceFluxes(const std::vector<dstype>& recv)` | inject the interface fluxes provided by the other side |

The interface-setup parameters:

| Parameter | Meaning |
|---|---|
| `ncuext` | number of components of the *external* field received |
| `ncuint` | number of components of the *internal* field sent |
| `ibc` | boundary tag identifying the interface |
| `comperm` | interface flux component map |
| `offset` | component offset into the auxiliary `odg` (`v`) array where the exchanged field lands |

`ExasimPoint` is a plain `{ dstype x, y, z; }` (with `dstype` = `double` or
`float` per the build). The `send`/`recv` vectors are flat `dstype` arrays laid
out over the interface points and components.

## Sub-iteration: save and restore state

To iterate a coupling step to convergence before advancing, checkpoint the
solver, retry with updated interface data, and restore until converged:

| Method | Purpose |
|---|---|
| `SaveState(modelnumber)` | checkpoint the current solution |
| `RestoreState(modelnumber)` | roll back to the last checkpoint |
| `ClearSavedState(modelnumber)` | discard the checkpoint |

## Skeleton driver

A generic outer loop: discover the interface, then for each coupling step
exchange data and sub-iterate to convergence. The partner is abstract — replace
`exchange_with_partner` with however your program moves the flux vectors.

```cpp
#include <exasim/ExasimSolverSetup.hpp>
#include <exasim/builtinlibprovider.hpp>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;

    ExasimSolver solver;
    if (int err = InitializeExasimSolver(solver, argc, argv, comm)) return err;

    // Declare the interface on model 0 (boundary tag ibc, field layout).
    const int model = 0, ncuext = 1, ncuint = 1, ibc = 1, comperm = 0, offset = 0;
    solver.IntializeMeshInterface(model, ncuext, ncuint, ibc, comperm, offset, comm);

    // The other side needs our interface coordinates to produce matching data.
    std::vector<ExasimPoint> pts = solver.getInterfacePoints();
    /* exchange_with_partner(pts); */

    std::vector<dstype> send, recv;
    int start = 0, steps = 1;

    for (int step = 0; step < num_coupling_steps; ++step) {
        solver.SaveState(model);

        for (int sub = 0; sub < max_subiters; ++sub) {
            /* recv = exchange_with_partner(send); */   // get the partner's fluxes
            solver.setInterfaceFluxes(recv);            // inject them

            solver.RunTimeDependent(start, steps);      // advance one window

            solver.getInterfaceFluxes(send);            // read ours back out
            if (/* converged */ true) break;
            solver.RestoreState(model);                 // else retry the window
        }

        solver.ClearSavedState(model);
        start += steps;
    }

    return solver.Finalize();
}
```
