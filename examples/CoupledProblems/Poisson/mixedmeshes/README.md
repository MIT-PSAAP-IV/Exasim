# Mixed triangle/quadrilateral Poisson coupling

This example couples independently meshed triangle and quadrilateral subdomains
with `interfacetype = 1`. The default uses triangles on model 1 and quads on
model 2. Set `triangleOnModel1 = false` in `pdeapp.m` to reverse the orientation.

Both models must use the same polynomial order and compatible one-dimensional
face quadrature. Their volume element type, element nodes, face nodes and
element-face permutations may differ. The current implementation is 2D and CPU
only; CUDA/HIP validation and 3D support remain open.

Run `pdeapp.m` from MATLAB. Set `curved = true` to make each family interpolate
the same analytic curved interface with its own nodes, producing a small
non-matching geometric gap.
