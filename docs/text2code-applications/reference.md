# Text2Code Reference Gateway

This page collects the canonical references for Text2Code applications.

## Complete Keyword References

| Topic | Reference |
| --- | --- |
| `pdeapp.txt` keys | [pdeapp.txt field reference](../reference/pdeapp.md) |
| `pdemodel.txt` grammar | [pdemodel.txt syntax reference](../reference/pdemodel.md) |
| Text2Code executable | [text2code reference](../reference/text2code.md) |
| Generated model semantics | [Model contract](../reference/model-contract.md) |
| CMake targets and package options | [CMake targets and options](../reference/cmake.md) |

## Reserved `pdemodel.txt` Output Names

```text
Flux Source Tdfunc Ubou Fbou FbouHdg
Sourcew Output Monitor Initu Initq Inituq Initw Initv
Avfield Fint EoS VisScalars VisVectors VisTensors
QoIvolume QoIboundary Fext
```

In default Exasim mode, `Flux`, `Source`, `Tdfunc`, `Ubou`, `Fbou`, and
`FbouHdg` must be present in `outputs`.

## Common `pdeapp.txt` Categories

| Category | Representative keys |
| --- | --- |
| App/model | `model`, `modelfile`, `meshfile`, `discretization`, `platform`, `gencode`, `gendatain` |
| Discretization | `porder`, `pgauss`, `torder`, `nstage`, `temporalscheme` |
| Dimensions | `ncu`, `ncv`, `ncw`, `nsca`, `nvec`, `nten`, `nvqoi`, `nsurf` |
| Solver | `NewtonIter`, `NewtonTol`, `GMRESiter`, `GMRESrestart`, `GMREStol`, `preconditioner` |
| Physics | `physicsparam`, `externalparam`, `tau`, `dt`, `time` |
| Sweep | `physicsparamcases`, `physicsparamwarmstart` |
| Boundary | `boundaryconditions`, `boundaryexpressions`, `curvedboundaries`, `periodicboundaries1` |
| Output | `saveParaview`, `saveSolFreq`, `saveSolOpt`, `saveResNorm`, `timestepOffset` |
| Parallel/HPC | `mpiprocs`, `neb`, `nfb`, `datapath`, `exasimpath` |

## Operators And Functions

The model DSL supports arithmetic expressions, vector/matrix indexing, loops,
function calls, common math functions, and matrix operations. See
[pdemodel.txt syntax](../reference/pdemodel.md#body-sublanguage) for the
current supported list.

## Example Locations

| Example | Path |
| --- | --- |
| Poisson 1D/2D/3D | `apps/poisson/` and `examples/Poisson/` |
| Navier-Stokes airfoil | `apps/navierstokes/naca0012steady/` |
| Navier-Stokes unsteady airfoil | `apps/navierstokes/naca0012unsteady/` |
| Slip cylinder | `examples/NavierStokes/slipcylinder/` |
| ALE Poisson | `examples/ALE/Poisson2d/` |

## See Also

- [Overview](index.md)
- [Workflow and troubleshooting](workflow.md)
- [Advanced topics](advanced.md)
