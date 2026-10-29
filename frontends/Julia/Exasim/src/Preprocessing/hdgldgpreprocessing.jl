"""
    hdgldgpreprocessing(config, manifest, outdir, executable)

Run shared CPU preprocessing and return `(H_datain, L_datain)`. Config must
already contain explicit native model metadata (including ncu/ncv/ncw), mesh
and field paths, and solver controls. Output must be a new absolute directory.
This does not generate model kernels or run the mixed solver.
"""
function hdgldgpreprocessing(config::AbstractString, manifest::AbstractString,
                                   outdir::AbstractString, executable::AbstractString)
    configpath, manifestpath, executablepath = realpath(config), realpath(manifest), realpath(executable)
    isabspath(outdir) && !ispath(outdir) ||
        throw(ArgumentError("rank-separated output must be a new absolute directory"))
    for path in (configpath, manifestpath, executablepath)
        isfile(path) || throw(ArgumentError("rank-separated input/executable must be a file: " * path))
    end
    run(Cmd([executablepath, configpath, manifestpath, String(outdir), "--native-metadata"]))
    isfile(joinpath(outdir, "selection.txt")) || error("rank-separated preprocessing omitted selection.txt")
    directories = (joinpath(outdir, "H"), joinpath(outdir, "L"))
    for (scheme, directory) in zip((1, 0), directories)
        values = open(joinpath(directory, "hdgldg.meta"), "r") do io
            strip(readline(io)) == "EXASIM_HDGLDG_REGION 1" ||
                error("rank-separated preprocessing produced invalid metadata")
            parse.(Int, split(readline(io)))
        end
        length(values) >= 2 && values[1] == scheme && values[2] >= 1 ||
            error("rank-separated preprocessing produced invalid region sizes")
        count = values[2]
        required = ["app.bin", "master.bin"]
        for rank in 1:count
            suffix = count > 1 ? string(rank) : ""
            append!(required, ["mesh" * suffix * ".bin", "sol" * suffix * ".bin",
                               "hdgldg" * suffix * ".txt"])
        end
        for filename in required
            isfile(joinpath(directory, filename)) ||
                error("rank-separated preprocessing omitted " * joinpath(directory, filename))
        end
    end
    return directories
end
