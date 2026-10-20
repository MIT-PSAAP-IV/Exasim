# Cook membrane

This standard nearly incompressible plane-strain benchmark deforms a tapered
48-by-60 membrane under a vertical traction. It uses a 10-by-10 quadrilateral
mesh, degree-three approximation, and the same Lamé parameters in all three
frontends.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
