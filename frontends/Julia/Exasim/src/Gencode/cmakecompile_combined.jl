# Build ONE solver executable that runs several generated models together
# (combined multi-PDE). Each pde in `pdes` must already have its kernels
# generated (Gencode.gencode) into its own per-model build dir and a distinct
# modelnumber (hence distinct modelid). Renders the frontend-app-combined
# templates into <builddir>/combined and builds against the installed Exasim.
# Mirrors the Python frontend's cmakecompile_combined.
function cmakecompile_combined(pdes)
    print("Compile combined multi-PDE Exasim app against the installed Exasim package...\n")

    prefix = install_prefix()
    p0 = pdes[1]
    builddir = joinpath(p0.builddir, "combined")
    mkpath(builddir)

    ids = Int[]
    kdirs = String[]
    for p in pdes
        p.exasimpath = prefix
        push!(ids, resolve_modelid(p))
        kd = joinpath(model_builddir(p), "kernels")
        isdir(kd) || error("No generated kernels at $kd; run Gencode.gencode(pde) first.")
        push!(kdirs, kd)
    end
    length(unique(ids)) == length(ids) ||
        error("combined models need distinct modelids (one per slot); got $ids")

    if p0.platform == "gpu"
        variant = p0.mpiprocs > 1 ? "gpumpi" : "gpu"
    else
        variant = p0.mpiprocs > 1 ? "cpumpi" : "cpu"
    end

    tmpl = frontend_app_combined_template_dir()
    subs = Dict(
        "EXASIM_VARIANT" => variant,
        "MODEL_IDS" => join(string.(ids), ", "),          # main.cpp: {100, 101}
        "MODEL_ID_LIST" => join(string.(ids), " "),       # CMake: IDS 100 101
        "KERNEL_DIRS" => join(["\"" * d * "\"" for d in kdirs], " "),  # KERNELS_DIRS
    )
    render(joinpath(tmpl, "CMakeLists.txt.in"), joinpath(builddir, "CMakeLists.txt"), subs)
    render(joinpath(tmpl, "main.cpp.in"), joinpath(builddir, "main.cpp"), subs)

    bdir = joinpath(builddir, "build")
    exe = joinpath(bdir, "exasimapp")
    cmake = cmake_command()
    cfg = `$cmake -S $builddir -B $bdir -DExasim_DIR=$(cmake_dir())`
    run(cfg)
    jobs = get(ENV, "JOBS", string(Sys.CPU_THREADS))
    run(`$cmake --build $bdir --parallel $jobs`)
    isfile(exe) || error("Build did not produce $exe.")
    return string(cfg)
end
