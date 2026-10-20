# Ionosphere Shell Test

This time-dependent five-equation ionosphere example is available through the
MATLAB, Python, and Julia frontends. The six-patch hexahedral shell extends
from 100 km to 500 km above a spherical Earth.

The example requires an external `input.nc` file containing the plasma and
background-field variables referenced by `pdeapp.*`. That data file is not
distributed with Exasim. Python requires `netCDF4`; Julia requires
`NCDatasets.jl`; MATLAB uses `ncread`.

After placing `input.nc` in this directory, run one frontend with its normal
Exasim environment configured:

```bash
matlab -batch "pdeapp"
python3 pdeapp.py
julia --project=../../../frontends/Julia/Exasim pdeapp.jl
```
