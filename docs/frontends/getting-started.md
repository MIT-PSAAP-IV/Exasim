# Getting Started With Frontends

This page shows the smallest useful frontend workflow and explains the files
that Exasim creates. For complete example applications, start from the
directories under `examples/`.

## Prerequisites

Install Exasim with frontend support:

```bash
cmake -S Exasim -B Exasim-build -DEXASIM_FRONTENDS=ON
cmake --build Exasim-build -j
cmake --install Exasim-build --prefix /path/to/exasim-prefix
```

The frontend setup scripts should set or discover `EXASIM_PREFIX`. If you use a
nonstandard installation, set it explicitly before running examples:

=== "MATLAB"

    ```matlab
    setenv('EXASIM_PREFIX', '/path/to/exasim-prefix')
    run('/path/to/exasim-prefix/share/exasim/matlab/exasim_setup.m')
    ```

=== "Python"

    ```bash
    export EXASIM_PREFIX=/path/to/exasim-prefix
    python3 pdeapp.py
    ```

=== "Julia"

    ```bash
    export EXASIM_PREFIX=/path/to/exasim-prefix
    julia pdeapp.jl
    ```

## Minimal Simulation Skeleton

The exact PDE model functions are problem dependent, but all frontends use the
same skeleton.

=== "MATLAB"

    ```matlab
    [pde, mesh] = initializeexasim();

    pde.model = "ModelD";
    pde.modelfile = "pdemodel";    % pdemodel.m on the MATLAB path
    pde.nd = 2;
    pde.ncu = 1;
    pde.nc = 3;                    % u plus q for a 2D LDG model
    pde.porder = 2;
    pde.physicsparam = [1.0];
    pde.saveParaview = 1;

    % Fill mesh.p, mesh.t, mesh.boundaryexpr, and mesh.boundarycondition here.
    [sol, pde, mesh, master, dmd, comstr, runstr] = exasim(pde, mesh);
    ```

=== "Python"

    ```python
    from exasim import initializeexasim, exasim

    pde, mesh = initializeexasim()
    pde["model"] = "ModelD"
    pde["modelfile"] = "pdemodel"   # pdemodel.py in the run directory
    pde["nd"] = 2
    pde["ncu"] = 1
    pde["nc"] = 3
    pde["porder"] = 2
    pde["physicsparam"] = [1.0]
    pde["saveParaview"] = 1

    # Fill mesh["p"], mesh["t"], boundary expressions, and boundary conditions.
    sol, pde, mesh = exasim(pde, mesh)
    ```

=== "Julia"

    ```julia
    using Exasim

    pde, mesh = Exasim.initializeexasim()
    pde.model = "ModelD"
    pde.modelfile = "pdemodel"      # pdemodel.jl in the run directory
    pde.nd = 2
    pde.ncu = 1
    pde.nc = 3
    pde.porder = 2
    pde.physicsparam = [1.0]
    pde.saveParaview = 1

    # Fill mesh.p, mesh.t, boundary expressions, and boundary conditions.
    sol, pde, mesh = Exasim.exasim(pde, mesh)
    ```

## Generated Files

By default, the frontends use two locations:

| Location | Purpose |
| --- | --- |
| `pde.datapath` | User-visible runtime data such as `datain/` and `dataout/`. Defaults to the current directory. |
| `pde.builddir` / `pde.buildpath` | Hidden generated-code and build area. Defaults to `.exasim/`. |

Typical output layout:

```text
<case>/
  datain/
    app.bin
    mesh.bin
    sol.bin
  dataout/
    out_np0.bin
    out_residualnorms0.bin
    vis*.vtu
  .exasim/
    model cache/build files
```

Parameter sweeps add case directories such as
`dataout/paramcase_0001/` and a sweep manifest. See
[Parameter sweeps](../usage-modes/parameter-sweeps.md).

## First Existing Examples

Run a language-matched example from its directory:

```bash
cd examples/Poisson/poisson2d
matlab -batch pdeapp
python3 pdeapp.py
julia pdeapp.jl
```

Not every example has all three language files. The repository contains
representative MATLAB, Python, and Julia examples across Poisson, advection,
Euler, Navier-Stokes, MHD, and other modules.

## See Also

- [Core data structures](data-structures.md)
- [Execution workflow](execution.md)
- [Postprocessing and visualization](postprocessing.md)
