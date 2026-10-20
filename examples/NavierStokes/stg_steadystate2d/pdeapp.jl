include(joinpath(dirname(@__DIR__), "_case_launcher.jl"))
main() = run_matlab(@__FILE__, @__DIR__, "pdeapp.m", joinpath(dirname(@__DIR__), "stg_steadystate2d_julia_run"))
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
