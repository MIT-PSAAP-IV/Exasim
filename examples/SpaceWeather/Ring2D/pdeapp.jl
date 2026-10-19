const REPOSITORY=dirname(dirname(dirname(@__DIR__)))
include(joinpath(REPOSITORY,"examples","NavierStokes","_case_launcher.jl"))
main()=run_matlab(@__FILE__,@__DIR__,"pdeapp.m",joinpath(dirname(@__DIR__),"Ring2D_julia_run"))
if abspath(PROGRAM_FILE)==abspath(@__FILE__); main(); end
