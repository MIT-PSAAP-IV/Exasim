# Orion flow and heat-transfer restart

The Orion workflow depends on checked-in binary fields and MATLAB-specific
axisymmetric mesh and solution-transfer routines. The Python and Julia entry
points run the canonical `pdeapp.m` in isolated copies so those fields and
their ordering are retained.

MATLAB is required. The launchers do not replace the coupled workflow with a
single-physics approximation.
