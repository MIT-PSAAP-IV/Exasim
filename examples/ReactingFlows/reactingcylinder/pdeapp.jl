const REPOSITORY=dirname(dirname(dirname(@__DIR__)))
include(joinpath(REPOSITORY,"examples","NavierStokes","_case_launcher.jl"))
main()=run_text2code(@__FILE__,joinpath(REPOSITORY,"apps","meshadaptivity","cylindermach8_nonequichem"),8,joinpath(@__DIR__,"julia_run"))
if abspath(PROGRAM_FILE)==abspath(@__FILE__); main(); end
