# Multi-Model Problems

Multi-model workflows couple distinct model definitions or subsystems. Exasim
supports these workflows through model definitions, interface/coupling
callbacks, shared arrays, external variables, and source/boundary terms.

The framework provides the hooks; the application controls the coupling
strategy.

## Coupling Styles

| Coupling style | Exasim implementation pattern |
| --- | --- |
| Monolithic | Pack all variables into one model's `u` and solve as one system. |
| Partitioned | Use multiple model definitions and exchange data through source, boundary, or external fields. |
| Source-term coupling | Put coupling terms in `source` / `Source`. |
| Boundary coupling | Use boundary/interface callbacks such as `Fint` and `Fext`. |
| External-variable coupling | Put another model's output in `v` / `odg`. |
| Auxiliary-state coupling | Use `w` and `sourcew` for model-specific state evolution. |

## Examples

- Flow model plus turbulence model.
- Flow model plus chemistry model.
- Flow model plus wave model.
- PDE model plus reduced-order model.
- Fluid model coupled to a thermal or material model.

## Multiple-Equation And Multi-Model Examples

The examples below show several distinct ways Exasim applications combine more
than one equation or model. They should not be treated as one universal
coupling API; each pattern has a different data-ownership and execution model.

| Example | Pattern | Main mechanism |
| --- | --- | --- |
| `examples/Poisson/multipleequations` | Simultaneous multi-model solve with solution transfer through `v`. | `pde{m}`, `modelnumber`, `vindx`. |
| `examples/Poisson/coupledproblem` | Two subdomain Poisson models coupled through an interface. | `interfacecondition`, `interfacefluxmap`. |
| `examples/MongeAmpere/square2d` | Staged Poisson + transport pipeline for mesh motion. | Reuse `mesh.vdg`, `mesh.udg`, and sequential calls. |
| `examples/MongeAmpere/square` | Reusable r-adaptivity driver with isolated model builds. | `combinedmodel`, `modelnumber`, staged Poisson/transport solves. |
| `examples/NavierStokes/nshtmach8` | Fluid-solid CHT with either partitioned transfer or multi-model interface setup. | `solutiontransfer_*`, `interfacecondition`, `interfacefluxmap`. |
| `examples/NavierStokes/nshtorion` | Partitioned Navier-Stokes / heat transfer iteration. | Boundary data transfer through `vdg`. |
| `examples/GSI/heatconcentration1d` | Monolithic two-equation heat/concentration system. | Pack both equations into one `u` vector. |
| `examples/ShallowWater/BickleyJet` | Alternative shallow-water model formulations. | Separate app/model files for different formulations. |

### `examples/Poisson/multipleequations`

This is the clearest checked-in example of a simultaneous multi-model frontend
run. The top-level `pdeapp.m` calls `pdeapp1.m` and `pdeapp2.m`, creating
cell-array entries `pde{1}`, `mesh{1}`, `pde{2}`, and `mesh{2}`. A single
`exasim(pde, mesh)` call then solves both models.

Model 1 is a scalar `ModelD` Poisson-like problem with
`pde{1}.modelnumber = 1` and `pde{1}.modelfile = "pdemodel1"`.

Model 2 is another scalar `ModelD` problem with
`pde{2}.modelnumber = 2` and `pde{2}.modelfile = "pdemodel2"`. It declares

```matlab
pde{2}.vindx = [1 1];
```

meaning that model 2 receives the first solution component of model 1 as an
external variable. In `pdemodel2.m`, that value is used through `v(1)`:

```matlab
function f = flux(u, q, w, v, x, t, mu, eta)
f = (mu+v(1))*q;
end
```

This illustrates external-variable coupling between separate model definitions:
model 1 owns the solution field, while model 2 consumes it through `v` / `odg`.

### `examples/Poisson/coupledproblem`

`examples/Poisson/coupledproblem` demonstrates interface coupling between two
Poisson subdomain models. The top-level `pdeapp.m` assigns

```matlab
pde{1}.modelnumber = 1;
pde{2}.modelnumber = 2;
mesh{1}.interfacecondition = [0;1;0;0];
mesh{2}.interfacecondition = [0;0;0;1];
pde{1}.interfacefluxmap = [1];
pde{2}.interfacefluxmap = [1];
```

The nonzero entries of `interfacecondition` mark which boundary of each
subdomain participates in the interface. The `interfacefluxmap` selects which
flux component is exchanged or constrained at that interface. This is a
boundary/interface-coupled multi-model pattern rather than a `vindx`
external-field pattern.

### `examples/MongeAmpere/square2d`

The `square2d` Monge-Ampere example is a staged multiple-equation workflow for
mesh adaptation:

1. `pdeapp_poisson.m` solves a `ModelD` Poisson-type equation for a potential.
2. The gradient components of that solution are stored as a velocity field:
   `mesh.vdg = sol(:,2:3,:)`.
3. `pdeapp_transport.m` solves a transport equation using that external
   velocity field.
4. The transport solution at final time becomes the moved mesh coordinate.
5. The transport solve is repeated for the \(x\)- and \(y\)-coordinate
   components.

This is not a simultaneous multi-model solve. It is a sequential PDE pipeline
where one equation produces an external field used by the next equation.

### `examples/MongeAmpere/square`

The `square` Monge-Ampere example wraps the same idea in `radaptivity.m`. It
uses `pde.combinedmodel = true` and assigns a separate model number for the
transport solve:

```matlab
pdet = pde;
pdet.modelnumber = 1;
pdet.modelfile = "pdemodel_transport";
```

The Poisson stage computes a mesh-density potential and stores several external
fields in `mesh.vdg`, including density data, density gradients, the recovered
velocity field, and a geometric Jacobian field. The transport stage then
advects mesh coordinates using those fields.

This example is useful when documenting staged model pipelines with isolated
build/data products rather than a single monolithic PDE.

### Monolithic Multiple-Equation Systems

Not every multiple-equation problem needs multiple model objects. Several
examples pack multiple PDEs into one `u` vector and solve them monolithically.
For example, `examples/GSI/heatconcentration1d` defines a two-component
`ModelD` system:

- `u(1)` is temperature.
- `u(2)` is concentration.
- `q(1)` and `q(2)` are the corresponding 1D gradients.
- `flux` returns heat and concentration diffusion terms.
- `source` couples the two equations through temperature-dependent chemistry.

This is the right pattern when the equations are strongly coupled and should be
assembled and solved as one nonlinear system.

### Alternative Formulation Families

Some directories contain multiple app/model files for related formulations
rather than a single coupled run. For example, `examples/ShallowWater/BickleyJet`
has separate `pdeapp*.m` / `pdemodel*.m` files for different shallow-water
formulations. These examples are useful for comparing model formulations, but
they are not necessarily multi-model couplings unless the top-level script
passes multiple `pde{m}` objects to one `exasim(...)` call or explicitly
transfers fields between solves.

## Practical Guidance

- Prefer a monolithic packed state when the coupling is strong and the
  nonlinear solve should see all variables together.
- Prefer `v` or source-term coupling when one field is prescribed or lagged.
- Use clear model numbers and output directories for multi-model runs.
- Document which model owns each state variable and which model only consumes
  it.
- Validate conservation across interfaces when using boundary coupling.
- Use `vindx` when one model needs selected solution components from another
  model as external variables.
- Use `interfacecondition` and `interfacefluxmap` when the coupling is imposed
  through a mesh boundary or subdomain interface.
- Use staged preprocessing/execution when a later equation depends on a field
  produced by an earlier equation but does not need simultaneous nonlinear
  coupling.
