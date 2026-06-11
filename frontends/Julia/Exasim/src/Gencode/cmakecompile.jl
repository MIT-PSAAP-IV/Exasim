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

    # Hash the model inputs (kernels + rendered app sources + the install);
    # if nothing changed since the last successful build, skip cmake entirely.
    stamp = joinpath(bdir, ".exasim_model_hash")
    # Templates rather than rendered files: rendered sources embed absolute
    # paths, which would make the digest directory-specific.
    digest = modelhash(kernels,
                       [joinpath(tmpl, "CMakeLists.txt.in"),
                        joinpath(tmpl, "main.cpp.in")],
                       prefix, variant, pde.modelid)
    if isfile(exe) && isfile(stamp) && read(stamp, String) == digest
        print("Model unchanged (hash match); skipping build.\n")
        return string(cfg)
    end

    # Per-user model cache: the relocatable (libfrontend_model, exasimapp)
    # pair built for this modelid+digest by any earlier app run.
    cachedir = joinpath(cacheroot(), string(pde.modelid), digest)
    cached = cachefiles(cachedir)
    if cached !== nothing
        mkpath(bdir)
        for src in cached
            cp(src, joinpath(bdir, basename(src)); force=true)
        end
        open(stamp, "w") do f
            write(f, digest)
        end
        print("Model cache hit ($cachedir); skipping build.\n")
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

    # Populate the cache for other app directories / future runs.
    libs = filter(n -> startswith(n, "libfrontend_model."), readdir(bdir))
    if !isempty(libs)
        mkpath(cachedir)
        for name in vcat(libs, ["exasimapp"])
            cp(joinpath(bdir, name), joinpath(cachedir, name); force=true)
        end
    end
    return string(cfg)
end

# Per-user cache of built model libraries (EXASIM_CACHE_DIR overrides).
cacheroot() = joinpath(get(ENV, "EXASIM_CACHE_DIR", joinpath(homedir(), ".exasim")), "cache")

# Return the cached (lib, exe) pair, or nothing if incomplete.
function cachefiles(cachedir)
    isdir(cachedir) || return nothing
    libs = filter(n -> startswith(n, "libfrontend_model."), readdir(cachedir))
    exe = joinpath(cachedir, "exasimapp")
    (!isempty(libs) && isfile(exe)) || return nothing
    return vcat([joinpath(cachedir, n) for n in libs], [exe])
end

# SHA-256 over the kernel set, the rendered app sources, and the identity of
# the Exasim install they build against.
function modelhash(kernelsdir, extra_files, prefix, variant, modelid)
    ctx = SHA.SHA256_CTX()
    SHA.update!(ctx, codeunits(string(variant, "|", modelid)))
    for name in sort(readdir(kernelsdir))
        SHA.update!(ctx, codeunits(name))
        SHA.update!(ctx, read(joinpath(kernelsdir, name)))
    end
    for path in extra_files
        SHA.update!(ctx, read(path))
    end
    SHA.update!(ctx, codeunits(string(prefix)))
    targets = joinpath(prefix, "lib", "cmake", "Exasim", "ExasimTargets.cmake")
    if isfile(targets)
        st = stat(targets)
        SHA.update!(ctx, codeunits(string(st.mtime, ":", st.size)))
    end
    return bytes2hex(SHA.digest!(ctx))
end
