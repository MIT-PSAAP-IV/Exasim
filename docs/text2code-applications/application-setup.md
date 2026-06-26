# Application Setup Guide

This page shows common `pdeapp.txt` setup patterns. Combine these with a
matching `pdemodel.txt` and mesh file.

## Steady-State Simulation

```text
dt = [0.0];
time = 0.0;
NewtonIter = 20;
NewtonTol = 1e-06;
saveSolFreq = 1;
```

Run:

```bash
/path/to/exasim-prefix/local/bin/text2code pdeapp.txt
cmake -S . -B build -DExasim_DIR=/path/to/exasim-prefix
cmake --build build
build/exasimapp datain/ dataout/out
```

The exact executable command depends on the app bundle or usage mode.

## Time-Dependent Simulation

```text
time = 0.0;
dt = [repeat(1e-4, 1000)];
torder = 2;
nstage = 2;
saveSolFreq = 20;
```

Use `saveSolFreq` to control saved solution and visualization frequency.

## Parameter Sweep

```text
physicsparam = [1.4, 0.72, 500.0];
physicsparamcases = [
  1.4, 0.72, 500.0;
  1.4, 0.72, 1000.0;
  1.4, 0.72, 1500.0;
  1.4, 0.72, 2000.0
];
physicsparamwarmstart = 1;
```

Text2Code writes `datain/physicsparamcases.bin`. The standalone executable
detects that file and runs all cases internally. Output directories follow the
same convention as frontend sweeps:

```text
dataout/
  physicsparam_sweep_manifest.txt
  paramcase_0001/
  paramcase_0002/
```

## MPI Run

```text
mpiprocs = 4;
platform = "cpu";
```

Use the MPI-enabled executable and launch it with the system MPI launcher:

```bash
mpirun -np 4 build/exasimapp datain/ dataout/out
```

Text2Code writes rank-local mesh and solution files when `mpiprocs > 1`.

## GPU Run

```text
platform = "gpu";
mpiprocs = 1;
```

Build the generated app with the GPU-enabled Exasim package. The CMake options
used by the app select CUDA or HIP, for example `-DEXASIM_CUDA=ON` or
`-DEXASIM_HIP=ON` when supported by the app CMake project.

## Restart-Oriented Configuration

Use saved solution files and consistent time-step offsets:

```text
saveSolFreq = 10;
timestepOffset = 1000;
time = 1.0;
```

Restart workflows depend on the executable/app wrapper used to read prior
solution files. Keep `datain`, `dataout`, `mpiprocs`, and mesh partitioning
consistent with the saved state.

## Postprocessing Workflow

Enable derived output:

```text
saveParaview = 1;
nsca = 2;
nvec = 1;
nvqoi = 2;
saveResNorm = 1;
```

Then run solve mode or explicit postprocess mode if the standalone app supports
it:

```bash
build/exasimapp postprocess datain/ dataout/out
```

See [Postprocessing](../usage-modes/postprocessing.md) for supported command
forms and output files.

## HPC Setup Checklist

- Use an installed Exasim prefix visible on compute nodes.
- Ensure `text2code` and generated executables use the same Exasim build.
- Match `platform` to the installed CPU/CUDA/HIP package.
- Keep `mpiprocs` consistent between preprocessing and execution.
- Put large `dataout/` directories on a filesystem intended for simulation
  output.
- Test with one rank and a coarse mesh before scaling.

## See Also

- [pdeapp.txt guide](pdeapp.md)
- [Workflow and troubleshooting](workflow.md)
- [Install: HPC](../install/hpc.md)
- [Parameter sweeps](../usage-modes/parameter-sweeps.md)
