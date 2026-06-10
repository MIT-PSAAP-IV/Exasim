# Compatibility shim for legacy pdeapps that include() this file after
# computing cdir/ii. The frontend now lives in the Exasim.jl package
# (frontends/Julia/Exasim); new code should `using Exasim` directly.
let srcdir = cdir[1:ii[end]] * "/frontends/Julia"
    if !(srcdir in LOAD_PATH)
        push!(LOAD_PATH, srcdir)
    end
end

import Exasim
# Re-export the historical bare module names for un-migrated pdeapps.
const Preprocessing = Exasim.Preprocessing
const Postprocessing = Exasim.Postprocessing
const Gencode = Exasim.Gencode
const Mesh = Exasim.Mesh
# Legacy pdeapps call Main.cmakecompile via Postprocessing.exasim; keep a
# top-level alias for any script that calls it directly.
const cmakecompile = Exasim.Gencode.cmakecompile

# Make external tools in common nonstandard prefixes visible without
# clobbering the user's PATH.
ENV["PATH"] = ENV["PATH"] * ":/usr/local/bin:/opt/local/bin:/opt/homebrew/bin";

print("==> Exasim Julia frontend (Exasim.jl package) ...\n");
