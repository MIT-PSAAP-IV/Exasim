# Postprocessing And Visualization

Frontend postprocessing covers loading binary solution files, plotting or
exporting visualization fields, writing ParaView files during backend
execution, and replaying postprocessing from saved solution data.

For the standalone backend implementation details, see
[Postprocessing](../usage-modes/postprocessing.md).

## Output Types

| Output | Controlled by | Notes |
| --- | --- | --- |
| Saved solution binaries | `saveSolFreq`, `saveSolOpt`, `dataoutpath` | Used by `fetchsolution`, restart, and postprocess mode. |
| Residual histories | `saveResNorm` | Written under `dataout` or case output directories. |
| ParaView VTK files | `saveParaview`, `nsca`, `nvec`, `nten`, visualization callbacks | Written by the backend during solve or postprocess mode. |
| Frontend visualization | `vis(...)`, `visscalars`, `visvectors`, `viselem`, `visfilename` | Language helper for visualization output. |
| QoI outputs | `nvqoi`, `nbqoi`, `qoivolume`, `qoiboundary` | Derived quantities from model callbacks. |

## Enable Backend ParaView Output

```matlab
pde.saveParaview = 1;
pde.nsca = 1;    % one scalar visualization field
pde.nvec = 1;    % one vector visualization field
pde.nten = 0;
```

The PDE model must define matching `visscalars`, `visvectors`, or `vistensors`
callbacks. If these counts are zero, no corresponding visualization field is
written.

## Frontend Visualization Helper

The `vis(...)` helper converts returned solution fields to visualization output
using frontend mesh and application metadata.

=== "MATLAB"

    ```matlab
    visfields = sol(:, :, :, end);
    vis(visfields, pde, mesh);
    ```

=== "Python"

    ```python
    from exasim import vis
    vis(sol, pde, mesh)
    ```

=== "Julia"

    ```julia
    Exasim.vis(sol, pde, mesh)
    ```

## Postprocess Execution Mode

Use `pde.executionmode = 1` to run the generated executable in postprocess mode
from a frontend.

```matlab
pde.executionmode = 1;
runstr = runcode(pde, 1);
```

This requires existing compatible `datain` and saved solution files. It is
useful when changing postprocessing options without rerunning the full solve.

## Parameter-Sweep Outputs

Parameter sweeps write outputs into case directories:

```text
dataout/
  physicsparam_sweep_manifest.txt
  paramcase_0001/
    physicsparam.txt
    out_np0.bin
  paramcase_0002/
    physicsparam.txt
    out_np0.bin
```

Postprocess each case by pointing `pde.dataoutpath` at the case directory or by
using the standalone sweep postprocessing workflow described in
[Parameter sweeps](../usage-modes/parameter-sweeps.md).

## Visualization In ParaView

Open the generated `.pvd`, `.pvtu`, or `.vtu` files from `dataout`. For MPI
runs, open the parallel file (`.pvtu` or `.pvd`) rather than individual rank
pieces when available.

Example:

```bash
paraview dataout/vis.pvd
```

## MATLAB Analysis

MATLAB examples often call utilities such as `fetchsolution`, `fetchresidual`,
`vis`, and problem-specific plotting scripts. These helpers use `master` and
`dmd` to interpret per-rank backend output.

```matlab
U = fetchsolution(pde, master, dmd, pde.dataoutpath);
```

## See Also

- [Execution workflow](execution.md)
- [Postprocessing usage mode](../usage-modes/postprocessing.md)
- [Parameter sweeps](../usage-modes/parameter-sweeps.md)
