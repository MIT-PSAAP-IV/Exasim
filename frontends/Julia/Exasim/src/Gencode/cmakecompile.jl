function cmakecompile(pde)
    print("Compile C++ Exasim code using cmake...\n")

    cdir = pwd()

    if pde.sharedbuild == 0
        sourcepath = joinpath(exasim_path(), "examples", "exasimfe")
        targetpath = joinpath(cdir, "exasim")
        isdir(targetpath) || mkpath(targetpath)

        for name in ("exasimfeapp.cpp", "CMakeLists.txt", "frontendprovider.cpp")
            cp(joinpath(sourcepath, name), joinpath(targetpath, name); force=true)
        end
    end

    sourcepath = pde.builddir
    buildpath = joinpath(pde.builddir, "build")
    isdir(buildpath) || mkpath(buildpath)

    exe = joinpath(buildpath, "exasimapp")
    isfile(exe) && rm(exe; force=true)

    args = String[
        "cmake",
        "-S", sourcepath,
        "-B", ".",
        "-D", "EXASIM_MPI=" * (pde.mpiprocs == 1 ? "OFF" : "ON"),
    ]

    if pde.platform == "gpu"
        push!(args, "-D", "EXASIM_CUDA=ON")
    elseif pde.platform == "hip"
        push!(args, "-D", "CMAKE_CXX_COMPILER=hipcc")
        push!(args, "-D", "EXASIM_HIP=ON")
    end

    pde.exasimpath = install_prefix()
    push!(args, "-D", "Exasim_DIR=" * pde.exasimpath)

    cfg = Cmd(Cmd(args); dir=buildpath)
    run(cfg)
    run(Cmd(`cmake --build . --target exasimapp --verbose`; dir=buildpath))

    return join(args, " ")
end
