# Multiphysics Problems

Multiphysics models combine several physical processes in one Exasim
application. Exasim supports this by allowing users to pack coupled state
variables into `u`, add auxiliary variables through `w`, add external fields
through `v`, and define coupled fluxes, sources, EOS closures, and boundary
terms.

This is a modeling pattern rather than a single automatic multiphysics solver:
the user defines the coupling law in the model callbacks.

## Coupling Mechanisms

| Mechanism | Exasim representation |
| --- | --- |
| Shared variables | Pack them into `u`. |
| Gradient-dependent coupling | Use `q` in `ModelD` or `ModelW`. |
| Auxiliary state | Use `w`, `initw`, and `sourcew`. |
| External fields | Use `v` / `odg` or `externalparam`. |
| Closure coupling | Use `eos` / `EoS`. |
| Boundary coupling | Use `ubou`, `fbou`, `fbouhdg`, `fint`, or `fext`. |
| Source-term coupling | Add coupled terms in `source` / `Source`. |

## Examples Of Multiphysics Formulations

- Reactive compressible flow.
- Plasma-flow coupling.
- Fluid-thermal coupling.
- Electromagnetic-fluid coupling.
- Multi-temperature nonequilibrium flow.

Relevant examples and modeling code include `examples/Euler/reacting_flow`,
`examples/ReactingFlows/reactingcylinder`, `examples/MHD/MagneticVortex`,
`examples/GSI/*`, and `frontends/Matlab/Modeling/CNS5air`.

## Conjugate Heat Transfer Examples

The `examples/NavierStokes/nshtorion` and
`examples/NavierStokes/nshtmach8` directories demonstrate conjugate heat
transfer (CHT) workflows that couple a compressible Navier-Stokes fluid model
to a solid heat-conduction model.

The coupled physics are:

| Subproblem | Exasim formulation | Representative files |
| --- | --- | --- |
| Fluid | `ModelD` Navier-Stokes with viscous heat flux and optional AV data in `v` / `odg`. | `pdeapp_ns.m`, `pdemodel_axialns.m`, `pdemodel_ns.m` |
| Solid | `ModelD` heat equation with thermal conductivity in `physicsparam`. | `pdeapp_ht.m`, `pdemodel_axialht.m`, `pdemodel_ht.m` |
| Interface | Heat flux from the fluid is imposed on the solid; solid temperature is imposed or transferred back to the fluid wall. | `solutiontransfer_ns2ht.m`, `solutiontransfer_ht2ns.m` |

### Partitioned CHT Iteration

`examples/NavierStokes/nshtorion/pdeapp.m` is a partitioned fluid-solid
iteration. The workflow is:

1. Solve or reuse the Navier-Stokes state on the fluid mesh.
2. Compute the wall heat flux from the fluid solution and trace data.
3. Transfer that heat flux to the solid heat mesh with
   `solutiontransfer_ns2ht.m`.
4. Store transferred interface data in `meshht.vdg`, which becomes the solid
   model's external variable field `v`.
5. Solve the solid heat equation.
6. Transfer the solid wall temperature back to the fluid mesh with
   `solutiontransfer_ht2ns.m`.
7. Store the transferred wall temperature in `mesh.vdg` for the fluid model.
8. Re-solve the fluid model and iterate until the fluid and solid interface
   temperatures are consistent.

In this pattern, Exasim supplies the two PDE solvers and the external-field
storage (`v` / `odg`). The coupling algorithm is written explicitly in MATLAB
by alternating preprocessing/execution of the two models and moving interface
data between meshes.

### Fully Coupled Interface Setup

`examples/NavierStokes/nshtmach8/pdeapp_fullycoupled.m` shows the multi-model
interface setup for a Mach-8 cylinder CHT problem. It creates two PDE objects:

| Model object | Meaning | Key settings |
| --- | --- | --- |
| `pde{1}` | Navier-Stokes fluid model. | `modelnumber = 1`, `mesh{1}.interfacecondition = [1;0;0]`, `pde{1}.interfacefluxmap = [4]`. |
| `pde{2}` | Heat-equation solid model. | `modelnumber = 2`, `mesh{2}.interfacecondition = [1;0;0]`, `pde{2}.interfacefluxmap = [1]`. |

The fluid-side `interfacefluxmap = [4]` selects the energy/heat-flux component
from the Navier-Stokes numerical flux. The solid-side `interfacefluxmap = [1]`
selects the scalar heat-equation flux. The nonzero entries in
`mesh{1}.interfacecondition` and `mesh{2}.interfacecondition` mark the
boundaries participating in the interface condition.

This setup is useful when the coupling is represented through Exasim's
multi-model interface machinery rather than an outer partitioned transfer loop.
The user still defines the interface physics through model boundary/interface
terms and consistent flux-component maps.

### Modeling Notes

- Both the fluid and heat subproblems use `ModelD` because the relevant fluxes
  depend on gradients: viscous/thermal fluxes in Navier-Stokes and conductive
  flux in the heat equation.
- The CHT data exchanged across the interface are not new primary unknowns in
  either single subproblem. They are communicated as boundary/interface data or
  as external variables in `v` / `odg`.
- The transfer scripts match DG nodes on the fluid and solid interface meshes
  before writing the received values into `vdg`.
- These examples are implementation patterns, not a universal CHT driver. Users
  should adapt the interface flux map, boundary IDs, and transfer logic to the
  variable ordering and mesh topology of their own CHT problem.

## Practical Strategy

1. Decide which variables are solved primary states and pack them into `u`.
2. Decide which gradients are needed; choose `ModelD` if diffusion or viscosity
   is present.
3. Put auxiliary dynamic or closure variables in `w`.
4. Put prescribed or externally supplied fields in `v`.
5. Keep EOS and transport closures in `eos` or small helper functions called by
   flux/source callbacks.
6. Document the variable ordering next to `initu`, `initw`, and the flux.
