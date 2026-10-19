# One-dimensional space-weather model

This workflow consumes `inputFiles.mat` and `euv.csv` and uses MATLAB-specific
atmospheric preprocessing and adapted-mesh routines. Python and Julia wrappers
execute `pdeapp.m` in isolated directories, preserving those data conventions.
MATLAB is required; no synthetic replacement atmosphere is introduced.
