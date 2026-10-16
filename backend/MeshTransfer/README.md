# Exasim MeshTransfer

Exasim::MeshTransfer is a stand-alone C++17/MPI/Kokkos library for
distributed high-order mesh-to-mesh field transfer. It does not require
changes to the Exasim solver and links only to Kokkos and, when enabled, MPI.

## Public API

The primary templates are
exasim::meshtransfer::MeshAdapter<Scalar, Index> and
exasim::meshtransfer::DistributedMeshTransfer<Scalar, Index>.

MeshAdapter owns the device inverse Vandermonde constructed from xpe.
Geometry and neighbor arrays are nonowning. DistributedMeshTransfer owns
and reuses its Kokkos/MPI workspace; target points and persistent
(erank,e,xi) arrays remain caller-owned aliases.

The transfer object provides evaluate_shape(), locate(), evaluate(),
transfer(), set_target_points(), set_num_components(), and reserve().

The flattened layouts are:

- xe[a + npe*d + npe*nd*elem]
- source[a + npe*c + npe*nc*elem]
- x[d + nd*i], xi[d + nd*i]
- target[c + nc*i]
- xpe[a + npe*d]
- neighborRank[face + nfe*elem]
- neighborElem[face + nfe*elem]

FOUND, LOCAL_NEXT, REMOTE, and OUTSIDE are the only geometric states.
OUTSIDE is terminal and evaluates the retained boundary-element polynomial
at its exterior reference coordinate.

## Build

    cmake -S backend/MeshTransfer -B build-mesh-transfer \
      -DMESHTRANSFER_ENABLE_MPI=ON
    cmake --build build-mesh-transfer -j
    ctest --test-dir build-mesh-transfer --output-on-failure

If an installed Kokkos package is unavailable, the standalone build uses
Exasim's vendored Kokkos source. GPU configurations require a CUDA- or
HIP-enabled Kokkos build and GPU-aware MPI because compact communication
buffers are passed directly to MPI.

## Numerical Notes

Tensor elements use shifted-Legendre modal bases; simplex elements use
total-degree monomials. Vandermonde inversion is setup-only host work with
partial pivoting and an infinity-norm condition estimate.

Exasim's existing point-locator small solve is host-only, and its Kokkos
small-matrix helpers launch whole-array kernels. MeshTransfer therefore
uses the same explicit 1x1, 2x2, and 3x3 determinant/cofactor formulas in
an allocation-free device function. No existing Exasim file is modified.

The current fixed device scratch limit is polynomial order 8 and 256 nodes
per element. Construction rejects larger cases explicitly. The validation
suite exercises orders 1 through 4; high-order equispaced simplex nodes
become progressively ill-conditioned and should be avoided in production.
