# Caltech cone restart workflow

This is a specialized finite-rate-chemistry restart workflow. It requires the
external files `solp2.mat`, `iter7final.mat`, and `fixed_av.mat`, plus the
MATLAB `CNS5air` chemistry library; those inputs are not distributed here.

`pdeapp.py` and `pdeapp.jl` preserve the canonical calculation by launching
`pdeapp.m` in an isolated directory. They fail clearly when the required
restart or chemistry inputs are absent. No reduced chemistry model is used.
