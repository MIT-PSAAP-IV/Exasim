# Glossary

| Term | Meaning |
| --- | --- |
| ABI | Application binary interface. Exasim uses `ExasimDriverABI` to connect model providers to the backend runtime. |
| ALE | Arbitrary Lagrangian-Eulerian formulation flag. |
| AV | Artificial viscosity used for shock capturing/stabilization. |
| Backend | C++ runtime implementing preprocessing, discretization, solvers, GPU/MPI, and output. |
| Built-in library | Installed library of predefined model providers. |
| DMD | Domain decomposition metadata used for MPI execution. |
| EOS | Equation of state or closure callback. |
| Frontend | MATLAB, Python, or Julia user interface and preprocessing layer. |
| HDG | Hybridizable discontinuous Galerkin discretization. |
| LDG | Local discontinuous Galerkin discretization. |
| ModelC | Convection-type model formulation without a `q + grad(u)` equation. |
| ModelD | Diffusion/mixed formulation with auxiliary gradient variables. |
| ModelW | Wave-type formulation with time-dependent auxiliary wave variables. |
| MPI | Message Passing Interface for distributed-memory execution. |
| `pde` | Frontend application/configuration structure. |
| `pdeapp.txt` | Text2Code application configuration file. |
| `pdemodel.txt` | Text2Code symbolic model-definition file. |
| `physicsparam` | Runtime vector of physical/model parameters. |
| Parameter sweep | Multiple simulations over several `physicsparam` vectors. |
| Postprocessing | Reading saved solution data and writing derived outputs such as VTK or QoI. |
| Provider | Code module that supplies model callbacks through the ABI. |
| QoI | Quantity of interest. |
| Text2Code | Exasim tool that generates solver inputs and model code from text files. |
| `u` | Primary solution variables. |
| `q` | Auxiliary gradient/wave variables depending on model type. |
| `v` / `odg` | External or other-DG field. |
| `w` / `wdg` | Auxiliary state variables. |
| `uhat` | HDG trace variable. |

## Related Pages

- [Physics Models](../physics-models/index.md)
- [Theory](../theory/index.md)
- [Internals](../internals/index.md)
