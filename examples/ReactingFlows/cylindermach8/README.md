# Finite-rate reacting Mach-8 cylinder

The chemistry model depends on the generated five-species-air implementation,
which has no native Python or Julia symbolic frontend. The entry points copy
`apps/meshadaptivity/cylindermach8_nonequichem`, run `text2code`, and execute
the eight-rank application. Set `EXASIM_PREFIX` before running.
