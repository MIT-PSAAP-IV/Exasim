# Backend Material Database

This note documents the PSAAP backend material-database path.  The frontend
packages an optional material table as:

```text
datain/materialdatabase.bin
```

The backend reads that file during normal input initialization.  If the file is
absent, all material-database metadata remain zero and all material-database
pointers remain null, so existing applications run unchanged.

## Binary layout

`materialdatabase.bin` is the compact table-form material database.  It stores
only `Float64` values:

```text
nstate nprop n1 n2 n3
state_1 ... state_nstate prop_1 ... prop_nprop
...
```

The first five values are integer-valued header entries stored as doubles:

- `nstate`: number of material-state coordinates, with `1 <= nstate <= 3`;
- `nprop`: number of stored properties, with `nprop >= 1`;
- `n1`, `n2`, `n3`: structured grid counts in the three possible state-space
  directions.

Inactive dimensions must have count one: `n2=n3=1` for one-dimensional tables,
and `n3=1` for two-dimensional tables.  The remaining records may be in any
order; the backend sorts and reconstructs the tensor-product table.

The binary file does not store appstruct arrays, material meshes, interpolation
metadata, names, units, equation-of-state identifiers, or provider metadata.

## appstruct storage

After reading and validating the table, the backend stores the derived material
interpolation data directly in `appstructT`:

```text
materialdb_nstate
materialdb_nprop
materialdb_porder
materialdb_elemtype
materialdb_npe
materialdb_ne

materialdb_elementcounts
materialdb_ncgi
materialdb_gridoffset
materialdb_elemoffset

materialdb_statecoords
materialdb_propvalues
materialdb_gridcoords
materialdb_elemcoords
```

No persistent `MaterialMesh` object is introduced into solver APIs.  Runtime
code accesses material data through these `app.materialdb_*` fields.

## Polynomial-order selection

The backend chooses the largest supported tensor-product interpolation order
from `p = 5, 4, 3, 2, 1` such that every active state dimension satisfies:

```text
(N_is - 1) % p == 0
```

where `N_is` is the number of unique grid coordinates in state dimension `is`.
For the selected order,

```text
materialdb_elementcounts[is] = (materialdb_ncgi[is] - 1) / materialdb_porder
materialdb_npe = (materialdb_porder + 1)^materialdb_nstate
materialdb_ne  = product(materialdb_elementcounts)
```

Active dimensions require at least two grid points.  If no valid order exists,
the input is rejected with a clear material-database error.

## Coordinate arrays

The material database is logically structured but may be nonuniformly spaced.
The backend therefore preserves explicit coordinates rather than deriving a
constant spacing.

`materialdb_gridcoords` stores all unique sorted grid coordinates, concatenated
by state dimension.  `materialdb_gridoffset` gives the segment for each state
coordinate:

```text
materialdb_gridcoords[materialdb_gridoffset[is] : materialdb_gridoffset[is+1]]
```

`materialdb_elemcoords` stores element-boundary coordinates, selected directly
from the corresponding grid-coordinate segment every `materialdb_porder` points.
`materialdb_elemoffset` gives the segment for each state dimension:

```text
materialdb_elemcoords[materialdb_elemoffset[is] : materialdb_elemoffset[is+1]]
```

Point-to-element logic should use `materialdb_elemcoords` and
`materialdb_elemoffset`; it must not assume uniform spacing.

## Element-local layouts

The state coordinates and property values are stored in Exasim-style
column-major flattened arrays:

```text
materialdb_statecoords[a + npe * (is + nstate * e)]
materialdb_propvalues [a + npe * (ip + nprop  * e)]
```

where:

- `a` is the local tensor-product interpolation node;
- `is` is the material-state coordinate index;
- `ip` is the material-property index;
- `e` is the material element index.

The first material-state dimension varies fastest in both tensor-product local
node numbering and structured element numbering.

## Validation

The backend reader validates only generic table structure:

- valid `nstate`, `nprop`, and active/inactive dimensions;
- expected row and column counts;
- finite numerical values;
- unique state points;
- complete tensor-product sampling;
- strictly increasing sorted axes;
- valid polynomial-order partitioning.

It does not assume any material model, fluid, solid, density, energy,
temperature, pressure, or thermodynamic admissibility.  Material-model-specific
checks belong outside this generic reader.


## Runtime material-property evaluation

When a material database is present, `wEquation` interprets the global auxiliary
state as:

```text
wdg = [ordinary auxiliary variables, material properties]
```

The final `app.materialdb_nprop` components are derived material properties.
Only the leading

```text
ncwa = common.components.ncw - app.materialdb_nprop
```

components participate in the wave/DAE auxiliary-variable update and Newton
solve.  Generated `Sourcew`/`HdgSourcew` kernels keep their existing total-`ncw`
ABI, so the material-aware `wEquation` path stores those outputs in full-width
temporary buffers and compacts only the leading `ncwa` residual/Jacobian blocks
before calling the small dense Newton solves.

After the ordinary auxiliary variables are available, `MaterialstateDriver`
computes the independent material-state coordinates, and
`materialproperties_kokkos` interpolates the tabulated properties into the last
`app.materialdb_nprop` components of `wdg`.  The derivative path applies the
chain rule

```text
dstate/dudg = state_udg_partial + state_wdg_partial * dw_aux/dudg
dproperties/dudg = dproperties/dstate * dstate/dudg
```

Only the ordinary auxiliary variables enter the first contraction; material
properties are outputs and are not nonlinear w-equation unknowns.

The Kokkos material-property lookup uses `materialdb_elemcoords` and
`materialdb_elemoffset` as the authoritative element-boundary coordinates.
Interior intervals follow `[x_left, x_right)`, and the global upper boundary is
included in the final element.  Query points outside the tabulated coordinate
range abort with a material-property domain error instead of being clamped to a
boundary element.

## Tests

The focused backend tests are registered as:

```text
appstruct_materialdatabase
material_properties_nonuniform
wequation_material_split
```

They cover:

- default release behavior with no database;
- one-dimensional order selection for `N=11`, `N=13`, and `N=10`;
- multidimensional Cartesian table reconstruction;
- nonuniform coordinate preservation;
- three-dimensional tensor-product construction;
- duplicate state-point rejection;
- nonuniform material-property interpolation and analytic derivatives;
- material-property boundary handling, including the global upper boundary;
- full-`Sourcew` to compact-`ncwa` Jacobian extraction for the material-aware
  w-equation split.

Run them from a configured build with:

```bash
ctest --test-dir <build-dir> -R "appstruct_materialdatabase|material_properties_nonuniform|wequation_material_split" --output-on-failure
```
