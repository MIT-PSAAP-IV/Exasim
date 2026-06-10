# Build the solver executable for the generated model.
#
# Renders the installed frontend-app templates into the hidden pde.builddir
# and builds them against the installed Exasim package via find_package(Exasim).
# The gencode step must have written the kernel set to <builddir>/kernels
# first. Returns the configure command. Julia's `run` throws on a nonzero
# exit, so configure/build failures propagate.

function render(template_path, dest_path, subs)
    text = read(template_path, String)
    for (key, val) in subs
        text = replace(text, "@" * key * "@" => string(val))
    end
    # Write only on change so unchanged apps don't dirty mtimes (recompiles).
    if isfile(dest_path) && read(dest_path, String) == text
        return
    end
    open(dest_path, "w") do f
        write(f, text)
    end
end

function cmakecompile(pde)
    print("Compile C++ Exasim code against the installed Exasim package...\n")

    prefix = install_prefix()
    pde.exasimpath = prefix

    builddir = pde.builddir
    kernels = joinpath(builddir, "kernels")
    if !isdir(kernels)
        error("No generated kernels at $kernels; run Gencode.gencode(pde) first.")
    end

    if pde.platform == "gpu"
        variant = pde.mpiprocs > 1 ? "gpumpi" : "gpu"
    else
        variant = pde.mpiprocs > 1 ? "cpumpi" : "cpu"
    end

    tmpl = frontend_app_template_dir()
    subs = Dict(
        "EXASIM_VARIANT" => variant,
        "MODEL_ID" => pde.modelid,
        "KERNEL_DIR" => kernels,
    )
    render(joinpath(tmpl, "CMakeLists.txt.in"), joinpath(builddir, "CMakeLists.txt"), subs)
    render(joinpath(tmpl, "main.cpp.in"), joinpath(builddir, "main.cpp"), subs)

    bdir = joinpath(builddir, "build")
    exe = joinpath(bdir, "exasimapp")
    cfg = `cmake -S $builddir -B $bdir -DExasim_DIR=$(cmake_dir())`

    # Hash the model inputs; if nothing changed since the last successful
    # build, skip cmake entirely.
    stamp = joinpath(bdir, ".exasim_model_hash")
    digest = modelhash(kernels,
                       [joinpath(builddir, "CMakeLists.txt"),
                        joinpath(builddir, "main.cpp")])
    if isfile(exe) && isfile(stamp) && read(stamp, String) == digest
        print("Model unchanged (hash match); skipping build.\n")
        return string(cfg)
    end

    run(cfg)
    jobs = get(ENV, "JOBS", string(Sys.CPU_THREADS))
    run(`cmake --build $bdir --parallel $jobs`)

    if !isfile(exe)
        error("Build did not produce $exe.")
    end
    open(stamp, "w") do f
        write(f, digest)
    end
    return string(cfg)
end

# SHA-256 over the kernel set and the rendered app sources.
function modelhash(kernelsdir, extra_files)
    ctx = SHA.SHA256_CTX()
    for name in sort(readdir(kernelsdir))
        SHA.update!(ctx, codeunits(name))
        SHA.update!(ctx, read(joinpath(kernelsdir, name)))
    end
    for path in extra_files
        SHA.update!(ctx, read(path))
    end
    return bytes2hex(SHA.digest!(ctx))
end
