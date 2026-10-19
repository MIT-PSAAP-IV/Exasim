include(joinpath(dirname(@__DIR__), "_case_launcher.jl"))
const REPOSITORY = dirname(dirname(dirname(@__DIR__)))
main() = run_text2code(@__FILE__, joinpath(REPOSITORY, "apps", "shockcapturing", "cylindermach8"), 1, joinpath(@__DIR__, "julia_run"))
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
