# Hafnium/HfO2 heat and concentration transport

This GSI case models coupled temperature and phase-concentration transport
through a hafnium/hafnium-oxide layer. The MATLAB, Python, and Julia drivers
share the material constants, nondimensional scales, degree-three
discretization, 300-element mesh, and accelerated time-step sequence.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
