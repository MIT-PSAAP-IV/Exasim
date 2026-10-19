"""Julia launcher for the canonical reacting Sharp-B Text2Code case."""

const CASEDIR = @__DIR__
const REPOSITORY = dirname(dirname(dirname(CASEDIR)))
const SOURCE = joinpath(REPOSITORY, "apps", "navierstokes", "reactingsharpb2")

function exasim_prefix()
    candidates = haskey(ENV, "EXASIM_PREFIX") ? [ENV["EXASIM_PREFIX"]] : String[]
    append!(candidates, [joinpath(dirname(REPOSITORY), "exasim_install"),
                         joinpath(dirname(REPOSITORY), "Exasim-build", "install")])
    for candidate in candidates
        isfile(joinpath(candidate, "bin", "text2code")) && return candidate
    end
    error("Set EXASIM_PREFIX to an Exasim installation containing bin/text2code")
end

function main(run_directory=joinpath(CASEDIR, "julia_run"))
    isdir(run_directory) && rm(run_directory; recursive=true)
    mkpath(dirname(run_directory))
    cp(SOURCE, run_directory; force=true)
    for generated in ("datain", "dataout", "partition.bin")
        path = joinpath(run_directory, generated)
        ispath(path) && rm(path; recursive=true)
    end
    prefix = exasim_prefix()
    cd(run_directory) do
        run(`$(joinpath(prefix, "bin", "text2code")) pdeapp.txt`)
        run(`mpiexec -n 4 $(joinpath(prefix, "bin", "exasimapp")) pdeapp.txt`)
    end
    println("Julia launcher completed: ", joinpath(run_directory, "dataout"))
    return run_directory
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
