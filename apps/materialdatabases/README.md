# Material databases

`equilibriumAir5.dat` is a 129-by-129 equilibrium-air table for the neutral
species `N2`, `O2`, `NO`, `N`, and `O`. It uses Cantera NASA-9 thermodynamics,
mixture-averaged transport, equilibrium derivatives, and equilibrium (not
frozen) sound speed. Its energy coordinate includes formation and dissociation
energy in Cantera's native reference convention. See
[`equilibrium_air5/README.md`](equilibrium_air5/README.md) for the precise
model, domain, column order, ionization limit, reproduction procedure, and
validation results.

`equilibriumAir5logdensitymutationpp.dat` has the same 161-by-161 coordinates
but stores Mutation++ conductivity as two separate columns:
`kappa_frozen` and `kappa_reactive`. Its first columns are
`xi e p T mu kappa_frozen kappa_reactive a_eq ...`, and the header is
`2 15 161 161 1`. It is generated independently with Mutation++ `v1.0.5`,
the built-in `air_5` mixture, NASA-9 thermodynamics, Gupta-Yos viscosity,
Wilke heavy conductivity, and equilibrium sound speed. 

`equilibriumAir5logdensityexasim.dat` is the Exasim-native CNS5air variant on
the same uniform 161-by-161 `(xi,e)` grid. It is generated directly with
`frontends/Matlab/Modeling/CNS5air/equilibrate.m` and
`transportcoefficients.m`. Thus its thermodynamics, five-species equilibrium
composition, Blottner/Wilke viscosity, and Eucken/Wilke-type conductivity are
consistent with Exasim's CNS5air model. A separate reactive-conductivity column
is copied from the Mutation++ extended table and placed immediately after the
CNS5air `kappa` column. See the equilibrium-air README for its hybrid transport
definition, generation command, derivative and sound-speed definitions,
validation, and limitations.
