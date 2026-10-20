# Rarefied slip-cylinder continuation

This workflow ramps Reynolds number and velocity/temperature accommodation
parameters over multiple restarts. Its wall-state model and staged control
logic are currently MATLAB-specific.

The Python and Julia entry points run the canonical `pdeapp.m` in isolated
directories. MATLAB is required, and all continuation stages are retained.
