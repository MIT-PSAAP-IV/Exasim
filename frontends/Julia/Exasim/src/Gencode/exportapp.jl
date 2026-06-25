# Package a relocatable "data transfer app" bundle.
#
# Mirrors the Python frontend's Gencode/exportapp.py. Collects everything the
# solver needs to run on any machine that has an Exasim install: the binary
# inputs (datain/, plus an empty dataout/), the generated kernel .cpp set, a
# relocatable CMakeLists.txt, main.cpp, a run.sh, a manifest, and a copy of the
# source model. The bundle depends on an Exasim install at run time
# (find_package(Exasim)) but carries no absolute paths, so it can be copied
# anywhere.
#
# Boundary conditions need no text form here: preprocessing has already
# resolved them into the binary datain. Requires preprocessing (datain) and
# gencode (kernels) to have run already.
#
# With build=true (default) the bundle is configured, built, and run in a
# throwaway scratch dir against the local Exasim install to verify it works
# before hand-off, leaving the shipped bundle pristine (no build/, empty
# dataout/).

# The Exasim solver variant implied by platform + process count.
function _variant(pde)
    if pde.platform == "gpu"
        return pde.mpiprocs > 1 ? "gpumpi" : "gpu"
    end
    return pde.mpiprocs > 1 ? "cpumpi" : "cpu"
end

# Mirror src into dst (replacing dst), preserving file contents.
function _copytree(src, dst)
    isdir(dst) && rm(dst; recursive=true)
    cp(src, dst)
end

function exportapp(pde, dest=nothing; build=true)
    if dest === nothing
        dest = pde.exportapp
    end
    if dest === nothing || isempty(dest)
        error("exportapp: no destination (set pde.exportapp or pass dest).")
    end
    dest = abspath(dest)

    builddir = pde.builddir
    kernels = joinpath(builddir, "kernels")
    if !isdir(kernels)
        error("No generated kernels at $kernels; run Gencode.gencode(pde) first.")
    end

    datain = joinpath(pde.datapath, "datain")
    if !isdir(datain)
        error("No datain at $datain; run Preprocessing.preprocessing(pde, mesh) first.")
    end

    print("Export Exasim data-transfer app to $dest ...\n")
    mkpath(dest)

    # 1. Runtime inputs + the output directory the solver writes into.
    _copytree(datain, joinpath(dest, "datain"))
    mkpath(joinpath(dest, "dataout"))
    _write_physicsparamcases_if_needed(pde, joinpath(dest, "datain"))

    # 2. The generated kernel .cpp set.
    _copytree(kernels, joinpath(dest, "kernels"))

    # 3 + 4. Relocatable CMakeLists.txt + main.cpp from the shared templates.
    # KERNEL_DIR resolves inside the bundle at configure time (not an absolute
    # path), which is what makes the bundle relocatable. The Exasim install is
    # the only external dependency, supplied on the target via Exasim_DIR.
    variant = _variant(pde)
    numpde = 1
    modelid = resolve_modelid(pde)
    tmpl = frontend_app_template_dir()
    subs = Dict(
        "EXASIM_VARIANT" => variant,
        "MODEL_ID" => string(modelid),
        "KERNEL_DIR" => "\${CMAKE_CURRENT_SOURCE_DIR}/kernels",
    )
    render(joinpath(tmpl, "CMakeLists.txt.in"), joinpath(dest, "CMakeLists.txt"), subs)
    render(joinpath(tmpl, "main.cpp.in"), joinpath(dest, "main.cpp"), subs)

    # 5. A copy of the source model, for reference / reproducibility.
    modelsrc = string(pde.modelfile) * ".jl"
    if isfile(modelsrc)
        cp(modelsrc, joinpath(dest, basename(modelsrc)); force=true)
    end

    # 5b. A text2code DSL file (pdemodel.txt) emitted from the symbolic model,
    # so the bundle can regenerate kernels via text2code without the Julia
    # frontend. Additive: if generation fails, warn and continue (the kernels
    # are already shipped in kernels/).
    try
        genpdemodel(pde, joinpath(dest, "pdemodel.txt"))
    catch exc
        @warn "pdemodel.txt generation skipped; the exported kernels/ are unaffected." exception=exc
    end

    # 6. run.sh + manifest with the provenance needed to rebuild elsewhere.
    _write_runscript(dest, pde, variant, numpde)
    _write_manifest(dest, pde, variant, numpde)
    _write_readme(dest, variant, numpde)

    if build
        _build_and_run(dest, pde, numpde)
    end

    print("Exported data-transfer app: $dest\n")
    return dest
end

# Write the shared standalone-sweep input file used by the C++ runtime.
function _write_physicsparamcases_if_needed(pde, datain)
    if !isdefined(pde, :physicsparamsweep) || isempty(pde.physicsparamsweep)
        return
    end
    cases = _physicsparam_sweep_cases(pde.physicsparamsweep, length(vec(pde.physicsparam)))
    if size(cases, 1) <= 1
        return
    end
    open(joinpath(datain, "physicsparamcases.bin"), "w") do f
        write(f, Float64[size(cases, 1), size(cases, 2)])
        write(f, vec(permutedims(Float64.(cases))))
    end
end

function _physicsparam_sweep_cases(spec, nparam)
    if spec isa Dict
        if haskey(spec, "samples")
            return _physicsparam_sweep_cases(spec["samples"], nparam)
        elseif haskey(spec, :samples)
            return _physicsparam_sweep_cases(spec[:samples], nparam)
        elseif haskey(spec, "values")
            return _physicsparam_sweep_cases(spec["values"], nparam)
        elseif haskey(spec, :values)
            return _physicsparam_sweep_cases(spec[:values], nparam)
        elseif haskey(spec, "grid")
            return _physicsparam_grid_cases(spec["grid"], nparam)
        elseif haskey(spec, :grid)
            return _physicsparam_grid_cases(spec[:grid], nparam)
        end
        error("physicsparamsweep Dict must contain samples, values, or grid.")
    elseif spec isa NamedTuple
        if haskey(spec, :samples)
            return _physicsparam_sweep_cases(spec.samples, nparam)
        elseif haskey(spec, :values)
            return _physicsparam_sweep_cases(spec.values, nparam)
        elseif haskey(spec, :grid)
            return _physicsparam_grid_cases(spec.grid, nparam)
        end
        error("physicsparamsweep NamedTuple must contain samples, values, or grid.")
    end

    if spec isa AbstractVector && !(eltype(spec) <: Number)
        cases = zeros(Float64, length(spec), nparam)
        for i in eachindex(spec)
            v = vec(Float64.(spec[i]))
            length(v) == nparam || error("physicsparamsweep case $i has $(length(v)) parameters; expected $nparam.")
            cases[i,:] = v
        end
    else
        arr = Float64.(spec)
        if ndims(arr) == 1
            if nparam == 1
                cases = reshape(arr, :, 1)
            elseif length(arr) == nparam
                cases = reshape(arr, 1, :)
            else
                error("physicsparamsweep vector has $(length(arr)) entries; expected $nparam.")
            end
        elseif ndims(arr) == 2
            cases = arr
        else
            error("physicsparamsweep must be numeric, a vector of vectors, a Dict, or a NamedTuple.")
        end
    end
    _validate_physicsparam_cases(cases, nparam)
    return cases
end

function _physicsparam_grid_cases(grid, nparam)
    length(grid) == nparam || error("physicsparamsweep.grid must contain one value vector per physics parameter.")
    rows = collect(Iterators.product((vec(Float64.(g)) for g in grid)...))
    cases = zeros(Float64, length(rows), nparam)
    for (i, row) in enumerate(rows)
        cases[i,:] = collect(row)
    end
    _validate_physicsparam_cases(cases, nparam)
    return cases
end

function _validate_physicsparam_cases(cases, nparam)
    size(cases, 2) == nparam || error("Each physicsparamsweep row must contain $nparam physics parameters.")
    all(isfinite, cases) || error("physicsparamsweep cases must contain finite numeric values.")
end

# The shell command line the bundle uses to launch the solver (MPI-aware).
function _run_command(pde, numpde, exe, datain, dataout)
    if pde.mpiprocs > 1
        return "\${MPIRUN:-mpirun} -np $(pde.mpiprocs) $exe $numpde $datain $dataout"
    end
    return "$exe $numpde $datain $dataout"
end

function _write_runscript(dest, pde, variant, numpde)
    runline = _run_command(pde, numpde, "\"\$here/build/exasimapp\"",
                           "\"\$here/datain/\"", "\"\$here/dataout/out\"")
    text = """#!/usr/bin/env bash
# Build and run this exported Exasim data-transfer app.
# Requires an Exasim install on this machine; point EXASIM_ROOT at its prefix
# (the directory that contains lib/cmake/Exasim/). For MPI variants, set
# MPIRUN if your launcher is not "mpirun".
#
# The bundle was exported for variant "$variant", but the kernels and datain
# are arch-independent: retarget to whatever the build machine has by setting
# EXASIM_VARIANT (cpu | cpumpi | gpu | gpumpi), e.g.
#   EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh
set -euo pipefail
here="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
: "\${EXASIM_ROOT:?Set EXASIM_ROOT to your Exasim install prefix}"
cmake -S "\$here" -B "\$here/build" \\
  -DExasim_DIR="\$EXASIM_ROOT/lib/cmake/Exasim" \\
  -DEXASIM_VARIANT="\${EXASIM_VARIANT:-$variant}"
cmake --build "\$here/build" --parallel
$runline
"""
    path = joinpath(dest, "run.sh")
    open(path, "w") do f
        write(f, text)
    end
    chmod(path, 0o755)
end

# JSON-escape a string value for the manifest (no JSON dep in this frontend).
function _json_str(s)
    s = replace(string(s), "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    return "\"" * s * "\""
end

function _write_manifest(dest, pde, variant, numpde)
    fields = [
        ("format", _json_str("exasim-data-transfer-app/1")),
        ("frontend", _json_str("julia")),
        ("modelid", string(pde.modelid)),
        ("variant", _json_str(variant)),
        ("platform", _json_str(pde.platform)),
        ("mpiprocs", string(pde.mpiprocs)),
        ("numpde", string(numpde)),
        ("hybrid", string(pde.hybrid)),
        ("porder", string(pde.porder)),
        ("built_against_prefix", _json_str(install_prefix())),
    ]
    lines = ["  " * _json_str(k) * ": " * v for (k, v) in fields]
    text = "{\n" * join(lines, ",\n") * "\n}\n"
    open(joinpath(dest, "manifest.json"), "w") do f
        write(f, text)
    end
end

function _write_readme(dest, variant, numpde)
    text = """# Exasim data-transfer app

Self-contained bundle for the generated model (variant `$variant`). It carries
the binary inputs, the generated kernels, and a relocatable CMake build, but
relies on an **Exasim install** on the machine where you build it.

## Contents
- `datain/`    binary solver inputs (mesh, master, app, solution); boundary
               conditions are already resolved here.
- `dataout/`   solver writes outputs here.
- `kernels/`   generated kernel .cpp set for this model.
- `CMakeLists.txt`, `main.cpp`   relocatable build of `exasimapp`.
- `run.sh`     build + run helper.
- `manifest.json`   provenance (model id, variant, process count, ...).

## Build & run
```sh
EXASIM_ROOT=/path/to/exasim/install ./run.sh
```
or manually:
```sh
cmake -S . -B build -DExasim_DIR=\$EXASIM_ROOT/lib/cmake/Exasim
cmake --build build --parallel
./build/exasimapp $numpde datain/ dataout/out
```
"""
    open(joinpath(dest, "README.md"), "w") do f
        write(f, text)
    end
end

# Configure, build, and run the bundle against the local Exasim install to
# verify it works before hand-off. The build and the run output go to a
# throwaway scratch directory so the shipped bundle stays pristine (source +
# datain + empty dataout); the scratch only proves the relocatable
# CMakeLists builds and the solver runs.
function _build_and_run(dest, pde, numpde)
    print("Verify exported app: build + run against the local Exasim install...\n")
    cmake = cmake_command()
    work = mktempdir(; prefix="exasim_export_verify.")
    try
        bdir = joinpath(work, "build")
        run(`$cmake -S $dest -B $bdir -DExasim_DIR=$(cmake_dir())`)
        jobs = get(ENV, "JOBS", string(Sys.CPU_THREADS))
        run(`$cmake --build $bdir --parallel $jobs`)

        exe = joinpath(bdir, "exasimapp")
        if !isfile(exe)
            error("Bundle build did not produce $exe.")
        end

        datain = joinpath(dest, "datain") * "/"
        dataout = joinpath(work, "out")
        if pde.mpiprocs > 1
            mpirun = pde.mpirun
            mpitxt = joinpath(bdir, "mpiexec.txt")
            if isfile(mpitxt)
                discovered = strip(read(mpitxt, String))
                isempty(discovered) || (mpirun = discovered)
            end
            cmd = `$mpirun -np $(pde.mpiprocs) $exe $numpde $datain $dataout`
        else
            cmd = `$exe $numpde $datain $dataout`
        end
        run(Cmd(cmd; dir=work))

        produced = filter(f -> startswith(f, "out"), readdir(work))
        if isempty(produced)
            error("Bundle verification run produced no output.")
        end
        print("Exported app verified (built and ran against the local install).\n")
    finally
        rm(work; recursive=true, force=true)
    end
end
