# Mixed triangle/quadrilateral Poisson coupling with a gap

This example extends `../mixedmeshes` by separating the two non-matching
interfaces by `0.02`. Model 1 occupies `[0, 0.49] x [0, 1]`, while model 2
occupies `[0.51, 1] x [0, 1]`. The default uses triangles on model 1 and
quadrilaterals on model 2. Set `triangleOnModel1 = false` in `pdeapp.m` to
reverse the orientation.

Both models use the manufactured solution

```text
u(x,y) = sin(pi*x) sin(pi*y).
```

The symmetry about `x=0.5` gives corresponding interface points equal solution
values and compatible outward normal fluxes even though their coordinates differ.
The non-matching interface solver uses polynomial extrapolation across the
gap; no elements fill the region `0.49 < x < 0.51`.

Both models use `interfacetype = 1` and must retain compatible one-dimensional
face spaces and quadrature. Their volume element families differ. The current
implementation is 2D and CPU only; CUDA/HIP validation and 3D support remain
open.

Run `pdeapp.m` from MATLAB. Set `curved = true` to add the same sinusoidal
interface displacement to both sides while retaining the constant `0.02` gap.
