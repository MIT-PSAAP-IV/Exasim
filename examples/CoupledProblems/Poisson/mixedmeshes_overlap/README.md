# Mixed triangle/quadrilateral Poisson coupling with an overlap

This example extends `../mixedmeshes` by overlapping the two non-matching
subdomains by `0.02`. Model 1 occupies `[0, 0.51] x [0, 1]`, while model 2
occupies `[0.49, 1] x [0, 1]`. The default uses triangles on model 1 and
quadrilaterals on model 2. Set `triangleOnModel1 = false` in `pdeapp.m` to
reverse the orientation.

Both models use the manufactured solution

```text
u(x,y) = sin(pi*x) sin(pi*y).
```

Symmetry about `x=0.5` gives corresponding interface points equal solution
values and compatible outward normal fluxes. Both models contain the strip
`0.49 < x < 0.51`, but their coupling boundaries remain the outer edge of
model 1 at `x=0.51` and the inner edge of model 2 at `x=0.49`.

Both models use `interfacetype = 1` and must retain compatible one-dimensional
face spaces and quadrature. Their volume element families differ. The current
implementation is 2D and CPU only; CUDA/HIP validation and 3D support remain
open.

Run `pdeapp.m` from MATLAB. Set `curved = true` to add the same sinusoidal
interface displacement to both sides while retaining the constant `0.02`
overlap. Figures 1 and 2 show the two models separately. Figure 3 shows the
combined solution and redraws model 1 edges in black and model 2 edges in red,
so neither topology is hidden in the overlap strip.
