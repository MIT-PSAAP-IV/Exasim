# Three-dimensional ISOQ Poisson problem

This case solves the same three-dimensional Laplace model and five boundary
classes as the MATLAB ISOQ example. Python and Julia use the checked-in
`isoq3d.geo` quarter-shell geometry because the legacy MATLAB mesh depends on
MATLAB-only `mkmesh_sphericalfrustum`, `rotate_mesh`, and `connectmesh`
utilities. The meshes therefore share the physical boundary topology but are
not node-identical. Gmsh is required for the Python and Julia drivers.

Run `pdeapp.m`, `python3 pdeapp.py`, or `julia pdeapp.jl` from this directory.
