#pragma once

#include <string>

// Read a binary snapshot file laid out as:
//   [npe, nc, ne] (stored as doubles),
//   followed by nsnapshots snapshots of size npe*nc*ne in column-major order
//   u[i + npe*c + npe*nc*e].
//
// Compute arithmetic Reynolds averages over the zero-based snapshot interval
// [stepoffsets, min(stepoffsets + nsteps, nsnapshots) - 1] using only the first
// five conservative Euler components:
//   0: rho, 1: rho*u, 2: rho*v, 3: rho*w, 4: rho*E.
//
// Write a binary output file laid out as:
//   [npe, 30, ne] (stored as doubles),
//   followed by npe*30*ne averaged fields in column-major order
//   avg[i + npe*c + npe*30*e].
//
// Output component order:
//   0  avg(rho)        10 avg(rho*u^2)   20 avg(u*w)
//   1  avg(rho*u)      11 avg(rho*v^2)   21 avg(v*w)
//   2  avg(rho*v)      12 avg(rho*w^2)   22 avg(rho^2)
//   3  avg(rho*w)      13 avg(rho*u*v)   23 avg(p^2)
//   4  avg(rho*E)      14 avg(rho*u*w)   24 avg(T^2)
//   5  avg(u)          15 avg(rho*v*w)   25 avg(rho*T)
//   6  avg(v)          16 avg(u^2)       26 avg(rho*T^2)
//   7  avg(w)          17 avg(v^2)       27 avg(rho*u*T)
//   8  avg(p)          18 avg(w^2)       28 avg(rho*v*T)
//   9  avg(T)          19 avg(u*v)       29 avg(rho*w*T)
//
// Temperature uses the Exasim Euler postprocessing convention:
//   T = p / ((gamma - 1) * rho).
//
// Extra input components c >= 5 are ignored. The implementation streams one
// snapshot at a time and throws std::runtime_error on invalid files, invalid
// starting offsets, invalid gamma, aliasing file paths, or zero density after
// applying fabs() to rho. If stepoffsets + nsteps exceeds the available snapshot
// count, the average is taken over stepoffsets through the last available snapshot.
void ReynoldsAverages3D(const std::string& fileout,
                        const std::string& filein,
                        int nsteps,
                        int stepoffsets,
                        double gamma);

// Convenience overload for averaging the last nsteps snapshots. It computes
// stepoffsets = nsnapshots - nsteps from the input file metadata, then delegates
// to the explicit-offset overload above.
void ReynoldsAverages3D(const std::string& fileout,
                        const std::string& filein,
                        int nsteps,
                        double gamma);
