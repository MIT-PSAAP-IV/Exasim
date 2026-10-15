# Coupled heat-transfer examples

Each example has MATLAB, Python, and Julia entry points:

```text
pdeapp.m
pdeapp.py
pdeapp.jl
```

The ports preserve the MATLAB polynomial and quadrature orders, physical
parameters, boundary/interface conditions, MPI rank split, nonlinear controls,
and the stand-alone fluid precursor used to initialize the monolithic coupled
solve.

Run an entry point from its case directory. For example:

```bash
matlab -batch pdeapp
python3 pdeapp.py
julia --project=../../../../frontends/Julia/Exasim pdeapp.jl
```

Python writes generated files under `python_run/`; Julia uses `julia_run/`.
MATLAB retains the historical `datain*` and `dataout*` locations.

After running all three frontends, compare their owned DG solutions with:

```bash
python3 ../verify_frontends.py
```

or run `verify_frontends.py` from this directory to compare all five cases.
The verifier removes the backend output header and matches elements and local
nodes by physical coordinates, so partition or node permutations are not
misreported as numerical errors. Independently generated non-matching Gmsh
meshes are compared using mesh-independent field statistics.

The Mach-8 cylinder cases agree with MATLAB to approximately `1e-12` relative
L2 after geometric matching. The ISOQ shock-flow cases show larger
frontend-to-frontend differences because their precursor and coupled solves
stop at the existing nonlinear tolerances (the matching solve reaches its
12-step cap); their residual histories, donor-search geometry, and physical
solution ranges should be compared as well as the final field norms.
