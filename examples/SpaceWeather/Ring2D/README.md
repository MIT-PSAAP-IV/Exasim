# Ring2D space-weather model

The case reads `euv.csv` and uses MATLAB-specific atmospheric initialization,
EUV source, scale-height mesh, and diagnostics. Python and Julia entry points
run the canonical workflow from isolated copies so the tabulated forcing and
mesh construction remain unchanged. MATLAB is required.
