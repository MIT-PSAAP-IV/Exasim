# Internal linear-elasticity models

This directory contains the 2D and 3D HDG operators used for pseudo-elastic
mesh movement. They solve `-div(sigma(u)) + f = 0`, with spatially varying
Lamé fields stored as `v = (mu, lambda, f)` and Exasim's `q = -grad(u)`
convention.

`pdeappN.txt` and `pdemodelN.txt` are the canonical Text2Code sources. When
Text2Code is available, CMake regenerates dimension-specific model headers and
builds the archive for the selected CPU, CUDA, or HIP backend. The handwritten
`linear_elasticity_model.hpp` remains a bootstrap fallback and implements the
same contract.

The mathematical sources are installed under
`share/exasim/models/LinearElasticity`.
