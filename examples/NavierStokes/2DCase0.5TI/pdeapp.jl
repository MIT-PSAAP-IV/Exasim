include(joinpath(dirname(@__DIR__), "_case_launcher.jl"))
main() = run_matlab(@__FILE__, @__DIR__, "pdeapp2d.m", joinpath(dirname(@__DIR__), "2DCase0.5TI_julia_run"))
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
