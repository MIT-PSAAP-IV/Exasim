# Two-dimensional seawall

This example computes the plane-strain deformation of a seawall cross-section
under body loading and boundary tractions. All frontends read the tracked Gmsh
mesh, use linear elements, and apply the same five boundary labels.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```

Regenerating the mesh requires Gmsh; running with the tracked mesh does not.
