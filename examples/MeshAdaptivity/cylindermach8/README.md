# Mach-8 cylinder mesh adaptivity

`pdeapp_frontend.m` implements the MATLAB reference continuation/adaptation
loop. `pdeapp_backend.m`, `pdeapp_backend.py`, and `pdeapp_backend.jl` exercise
the equivalent backend loop with the same 51-by-32 curved quadrilateral mesh,
second-order HDG discretization, physical parameters, initial condition, AV
continuation, and mesh-adaptivity settings.

Run the three backend drivers from this directory:

```bash
matlab -batch "pdeapp_backend"
/usr/bin/python3 pdeapp_backend.py
julia --startup-file=no pdeapp_backend.jl
```

The drivers use `backend_run`, `python_backend_run`, and `julia_backend_run`,
respectively. They disable `EXASIM_MESHADAPT_VERIFY`, so normal runs write only
the standard solution files and the final `out_meshadapt_xdg_np0.bin` adapted
coordinates rather than per-iteration diagnostic files.

After all three runs complete, compare their initial data, final adapted mesh,
flow solution, and AV fields with:

```bash
/usr/bin/python3 compare_backend_frontends.py
```

The comparison fails if any relative L2 error exceeds `1e-6`.
