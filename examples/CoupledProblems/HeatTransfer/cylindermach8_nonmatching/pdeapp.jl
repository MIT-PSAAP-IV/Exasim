repository=dirname(dirname(dirname(dirname(@__DIR__)))); pushfirst!(LOAD_PATH,joinpath(repository,"frontends","Julia","Exasim")); using Exasim
include(joinpath(dirname(@__DIR__),"cylinder_driver.jl"))
main()=run_cylinder_case(:nonmatching,@__DIR__)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
