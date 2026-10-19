# Five-species reacting cylinder flow

This example solves a nonequilibrium five-species air model (`N`, `O`, `NO`,
`N2`, and `O2`) around a cylinder. It couples species conservation, momentum,
total energy, a temperature material state, NASA-9 thermodynamics, and a
five-reaction Park chemistry model. Artificial viscosity is reduced through a
continuation sequence.

The Python and Julia ports contain local translations of the MATLAB chemistry
kernels in `reacting_kernels.py` and `reacting_kernels.jl`; no external
chemistry package is required. All three drivers use a curved 80-by-80,
`p = 3` quadrilateral mesh and target an NVIDIA GPU.

```bash
matlab -batch "run('pdeapp.m')"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```

The full continuation run is computationally expensive. Validate code
generation and chemistry expressions before scheduling the complete solve.
