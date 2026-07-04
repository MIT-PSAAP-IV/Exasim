# References

This page lists background references for the numerical methods used by Exasim.
It is not a complete bibliography, but it provides starting points for DG, LDG,
HDG, implicit time integration, GMRES, preconditioning, and performance-portable
GPU implementation.

## DG And LDG

- B. Cockburn and C.-W. Shu, "The Runge-Kutta discontinuous Galerkin method for
  conservation laws V: multidimensional systems", *Journal of Computational
  Physics*, 1998.
- B. Cockburn and C.-W. Shu, "The local discontinuous Galerkin method for
  time-dependent convection-diffusion systems", *SIAM Journal on Numerical
  Analysis*, 1998.
- B. Cockburn, G. E. Karniadakis, and C.-W. Shu, editors, *Discontinuous
  Galerkin Methods: Theory, Computation and Applications*, Springer, 2000.
- N. C. Nguyen, S. Terrana, and J. Peraire, "Large-Eddy Simulation of
  Transonic Buffet Using Matrix-Free Discontinuous Galerkin Method",
  *AIAA Journal*, 60(5), 3060-3077, 2022.
  [https://doi.org/10.2514/1.j060459](https://doi.org/10.2514/1.j060459).

## HDG

- B. Cockburn, J. Gopalakrishnan, and R. Lazarov, "Unified hybridization of
  discontinuous Galerkin, mixed, and continuous Galerkin methods for
  second-order elliptic problems", *SIAM Journal on Numerical Analysis*, 2009.
- B. Cockburn, J. Gopalakrishnan, and F.-J. Sayas, "A projection-based error
  analysis of HDG methods", *Mathematics of Computation*, 2010.
- N. C. Nguyen, J. Peraire, and B. Cockburn, "An implicit high-order
  hybridizable discontinuous Galerkin method for linear convection-diffusion
  equations", *Journal of Computational Physics*, 2009.
- N. C. Nguyen and J. Peraire, "Hybridizable discontinuous Galerkin methods for
  partial differential equations in continuum mechanics", *Journal of
  Computational Physics*, 231(18), 5955-5988, 2012.
  [https://doi.org/10.1016/j.jcp.2012.02.033](https://doi.org/10.1016/j.jcp.2012.02.033).

## Time Integration

- R. Alexander, "Diagonally implicit Runge-Kutta methods for stiff ODEs",
  *SIAM Journal on Numerical Analysis*, 1977.
- E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff
  and Differential-Algebraic Problems*, Springer.
- C. A. Kennedy and M. H. Carpenter, "Additive Runge-Kutta schemes for
  convection-diffusion-reaction equations", *Applied Numerical Mathematics*,
  2003.

## Krylov Solvers And Preconditioning

- Y. Saad and M. H. Schultz, "GMRES: A generalized minimal residual algorithm
  for solving nonsymmetric linear systems", *SIAM Journal on Scientific and
  Statistical Computing*, 1986.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, SIAM.
- A. Toselli and O. Widlund, *Domain Decomposition Methods: Algorithms and
  Theory*, Springer.
- A. Welter and N. Cuong Nguyen, "Preconditioning techniques for Hybridizable
  discontinuous Galerkin discretizations on GPU architectures", *Computer
  Methods in Applied Mechanics and Engineering*, 456, 118951, 2026.
  [https://doi.org/10.1016/j.cma.2026.118951](https://doi.org/10.1016/j.cma.2026.118951).
- P. Fernandez, N. C. Nguyen, and J. Peraire, "The hybridized Discontinuous
  Galerkin method for Implicit Large-Eddy Simulation of transitional turbulent
  flows", *Journal of Computational Physics*, 336, 308-329, 2017.
  [https://doi.org/10.1016/j.jcp.2017.02.015](https://doi.org/10.1016/j.jcp.2017.02.015).

## GPU And Performance Portability

- H. C. Edwards, C. R. Trott, and D. Sunderland, "Kokkos: Enabling manycore
  performance portability through polymorphic memory access patterns",
  *Journal of Parallel and Distributed Computing*, 2014.
- C. R. Trott, D. Lebrun-Grandie, D. Arndt, and contributors, Kokkos
  documentation and publications on performance-portable C++ programming.
- A. Welter and N. Cuong Nguyen, "Preconditioning techniques for Hybridizable
  discontinuous Galerkin discretizations on GPU architectures", *Computer
  Methods in Applied Mechanics and Engineering*, 456, 118951, 2026.
  [https://doi.org/10.1016/j.cma.2026.118951](https://doi.org/10.1016/j.cma.2026.118951).
- J. Vila-Pérez, R. L. van Heyningen, N.-C. Nguyen, and J. Peraire, "Exasim:
  Generating discontinuous Galerkin codes for numerical solutions of partial
  differential equations on graphics processors", *SoftwareX*, 20, 101212,
  2022.
  [https://doi.org/10.1016/j.softx.2022.101212](https://doi.org/10.1016/j.softx.2022.101212).

## Exasim And Related Applications

- D. S. Hoskin, R. L. van Heyningen, N. C. Nguyen, J. Vila-Pérez,
  W. L. Harris, and J. Peraire, "Discontinuous Galerkin methods for hypersonic
  flows", *Progress in Aerospace Sciences*, 146, 100999, 2024.
  [https://doi.org/10.1016/j.paerosci.2024.100999](https://doi.org/10.1016/j.paerosci.2024.100999).

## Exasim Internal Notes

Implementation-focused internal notes, including LDG implementation and
block-diagonal Jacobian derivations, are available as web pages:

- [LDG implementation deep dive](ldg-formulation.md)
- [Block-diagonal Jacobian](block-diagonal-jacobian.md)
