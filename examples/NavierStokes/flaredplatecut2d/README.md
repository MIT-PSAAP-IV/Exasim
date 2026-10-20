# Flared-plate cut-domain restart

The solve is initialized by transferring a prior flared-plate solution and
artificial-viscosity field onto a cut mesh. The required source solution lives
in the MATLAB workspace and is not distributed as a portable restart bundle.

The Python and Julia entry points therefore invoke `pdeapp.m` in isolation.
Prepare `sol1` and `av1` as described by the MATLAB workflow before running
either launcher.
