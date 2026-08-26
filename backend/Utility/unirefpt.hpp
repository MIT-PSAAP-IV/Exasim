#pragma once

#include <vector>

// unirefpt.hpp
//
// Uniform refinement of a LINEAR mesh -- its vertices p and element-to-vertex
// connectivity t -- subdividing each element into children `nref` times. This is
// the topological ("p and t") companion to the high-order DG-node refinement in
// refinemesh.{hpp,cpp}: this one produces the base mesh (new vertices + new
// connectivity); refinemesh then fills in the curved high-order nodes.
//
// Ports frontends/Matlab/Mesh/mkmesh/{uniref (tri), unirefquad (quad),
// uniref3d (tet)} and ADDS hex (which MATLAB lacks). Each element gains new
// vertices at its edge midpoints, and for tensor elements (quad/hex) at face
// centers and the cell center; shared entities between neighbouring elements are
// deduplicated TOPOLOGICALLY (keyed by the sorted tuple of the parent vertices
// that generate them), so the refined mesh is conforming exactly -- no
// coordinate-tolerance merge (fixmesh) needed.
//
// This is a serial host / preprocessing utility (irregular, hash-based dedup),
// unlike the data-parallel DG-node refinement.
//
// Layout (row-major, 0-based):
//   p : np*nd   vertex coords, p[i*nd + d]
//   t : nt*nv   connectivity,  t[e*nv + k]   (k-th corner vertex of element e)
// Element type is (nv, nd): tri (3,2), quad (4,2), tet (4,3), hex (8,3).
//
// p and t are refined IN PLACE (resized). Returns the number of children per
// element (4 for tri/quad, 8 for tet/hex); throws std::runtime_error for an
// unsupported (nv, nd).
int unirefpt(std::vector<double>& p, std::vector<int>& t, int nv, int nd, int nref);
