"""Isolated launch support for legacy Navier--Stokes examples."""

function repository_root(start::AbstractString)
    current = abspath(start)
    while true
        if isfile(joinpath(current, "CMakeLists.txt")) && isdir(joinpath(current, "frontends"))
            return current
        end
        parent = dirname(current)
        parent == current && error("could not locate the Exasim repository root")
        current = parent
    end
end

function exasim_prefix(repository::AbstractString)
    candidates = String[]
    haskey(ENV, "EXASIM_PREFIX") && push!(candidates, ENV["EXASIM_PREFIX"])
    append!(candidates, [
        joinpath(dirname(repository), "exasim_install"),
        joinpath(dirname(repository), "Exasim-build", "install"),
    ])
    for candidate in candidates
        if isfile(joinpath(candidate, "bin", "text2code")) &&
           isfile(joinpath(candidate, "bin", "exasimapp"))
            return candidate
        end
    end
    error("set EXASIM_PREFIX to an Exasim installation")
end

function fresh_copy(source::AbstractString, destination::AbstractString)
    isdir(destination) && rm(destination; recursive=true)
    mkpath(dirname(destination))
    cp(source, destination; force=true)
    for name in readdir(destination)
        if startswith(name, "datain") || startswith(name, "dataout") ||
           name in ("build", ".exasim", "python_run", "julia_run", "__pycache__")
            rm(joinpath(destination, name); recursive=true, force=true)
        end
    end
end

function run_text2code(case_file::AbstractString, source::AbstractString,
                       ranks::Integer, run_directory::AbstractString)
    repository = repository_root(dirname(case_file))
    fresh_copy(source, run_directory)
    prefix = exasim_prefix(repository)
    cd(run_directory) do
        run(`$(joinpath(prefix, "bin", "text2code")) pdeapp.txt`)
        run(`mpiexec -n $ranks $(joinpath(prefix, "bin", "exasimapp")) pdeapp.txt`)
    end
    return joinpath(run_directory, "dataout")
end

function run_matlab(case_file::AbstractString, source::AbstractString,
                    script::AbstractString, run_directory::AbstractString)
    fresh_copy(source, run_directory)
    escaped = replace(abspath(run_directory), "'" => "''")
    command = "cd('$escaped'); run('$script');"
    run(`matlab -batch $command`)
    return joinpath(run_directory, "dataout")
end
