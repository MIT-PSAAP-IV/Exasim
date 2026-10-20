# Non-matching 3D hex–tet Poisson coupling

This Phase 13 CPU diagnostic uses hexahedra in model 1 and tetrahedra in model
2. The B-M2 interface is a crossing mesh and is therefore diagnostic: Phase 13
does not introduce surface common refinement. Both models use equal order and
`pgauss = 2*porder`.

From MATLAB, run `pdeapp`. The driver uses the repository's manual Phase 13
harness and writes all generated data to a temporary directory.
