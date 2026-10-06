# Exasim CubicMovingMesh

`Exasim::CubicMovingMesh` is a standalone C++17/Kokkos header-only library
for a one-slab generalized cubic ALE mesh trajectory. It supports two- and
three-dimensional meshes and arbitrary stages `0 < c1 < c2`, including
`c2 > 1`.

The public class is

```cpp
exasim::cubicmovingmesh::CubicMovingMesh<Scalar, Index>
```

with `double` and `std::int32_t` defaults. Construct it with `(npe, nd, ne)`,
set the current slab with `setTimeSlab`, initialize or update the four cubic
states, and request deformation or derived geometry through the `calculate*`
methods in `include/cubic_moving_mesh.hpp`.

All Exasim-facing inputs and outputs use raw pointers. Read-only arrays are
accepted as `const Scalar*`, and outputs as `Scalar*`; callers do not need to
construct Kokkos view wrappers. On accelerator builds these pointers must
refer to memory accessible from `Kokkos::DefaultExecutionSpace`.

## Storage

All persistent arrays are one-dimensional Kokkos views in the default
execution space's memory space. The four mesh states have logical dimensions
`npe x (nd + nd*nd) x ne` and node-fastest address

```text
p + npe*c + npe*(nd + nd*nd)*e
```

The vector channels are `c=0,...,nd-1`. Matrix entry `(i,j)` is channel
`nd + i + nd*j`. Full deformation and velocity outputs use this packed layout;
matrix outputs omit the vector channels and use `p + npe*(i+nd*j) +
npe*nd*nd*e`. Jacobian outputs use `p + npe*e`.

The physical-coordinate input `X` is deliberately different: it contains
only `npe*nd*ne` values and uses

```text
p + npe*j + npe*nd*e
```

Deformation and velocity inputs are packed `npe*nc*ne` arrays. Undeformed
gradient channels are generated explicitly as the identity; `X` is never
accessed with the packed `nc` stride.

## Build and test

The local build is isolated from the rest of Exasim:

```bash
cmake -S backend/CubicMovingMesh -B build-cubic-moving-mesh \
  -DCUBIC_MOVING_MESH_BUILD_TESTS=ON
cmake --build build-cubic-moving-mesh -j
ctest --test-dir build-cubic-moving-mesh --output-on-failure
```

If no installed Kokkos is found, the build uses Exasim's vendored Kokkos with
the Serial backend. CUDA or HIP testing requires configuring this standalone
project against a Kokkos installation built for that backend.

## Numerical behavior

All geometry derives directly from the cubic mapping. Determinants, cofactors,
and inverses use explicit device-inline 2x2/3x3 algebra. Inverse evaluation
throws `CubicMovingMeshError` after detecting a non-positive or scale-relative
near-zero determinant. It does not clamp or regularize invalid geometry.
