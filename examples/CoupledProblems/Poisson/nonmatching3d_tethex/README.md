# Non-matching 3D tet–hex Poisson coupling

This Phase 13 CPU example uses tetrahedra in model 1, hexahedra in model 2,
and the nested A-M1 interface sequence. Both models use the same polynomial
order and `pgauss = 2*porder`; CUDA/HIP and mixed orders are intentionally not
enabled in Phase 13.

From MATLAB, run `pdeapp`. The driver uses the repository's manual Phase 13
harness and writes all generated data to a temporary directory.
