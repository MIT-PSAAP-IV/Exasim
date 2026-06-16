# Locate the Exasim install this package belongs to.
#
# The package is installed to <prefix>/share/exasim/julia/Exasim by
# `cmake --install`, so the install prefix is normally derived from the
# package location itself. Override with the EXASIM_PREFIX environment
# variable (e.g. for a source-tree checkout via Pkg.develop).

function _is_prefix(path::AbstractString)
    return isfile(joinpath(path, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
end

"""Return the Exasim install prefix, or throw with instructions."""
function install_prefix()
    env = get(ENV, "EXASIM_PREFIX", "")
    if !isempty(env)
        _is_prefix(env) && return env
        error("EXASIM_PREFIX=$env does not contain lib/cmake/Exasim/ExasimConfig.cmake")
    end

    here = @__DIR__                      # .../Exasim/src
    # Installed layout: <prefix>/share/exasim/julia/Exasim/src
    cand = abspath(joinpath(here, "..", "..", "..", ".."))
    _is_prefix(cand) && return cand
    # Source-tree layout: <repo>/frontends/Julia/Exasim/src with either a
    # local package under <repo>/local or the superbuild installed to
    # <repo>-build/install (sibling of the repo).
    repo = abspath(joinpath(here, "..", "..", "..", ".."))
    for cand in (joinpath(repo, "local"),
                 joinpath(dirname(repo), basename(repo) * "-build", "install"),
                 joinpath(repo, "build", "install"))
        _is_prefix(cand) && return cand
    end
    error("Cannot locate an Exasim install. Build and install Exasim " *
          "(cmake -S <repo> -B <repo>-build && cmake --build <repo>-build && " *
          "cmake --install <repo>-build) and/or set EXASIM_PREFIX to the install prefix.")
end

cmake_dir() = joinpath(install_prefix(), "lib", "cmake", "Exasim")
frontend_app_template_dir() = joinpath(cmake_dir(), "frontend-app")
text2code_path() = joinpath(install_prefix(), "bin", "text2code")

# Common locations a GUI launchd PATH omits; probed by absolute path.
const _TOOL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin")

# Locate an executable on this machine: PATH first, then common dirs.
function _find_tool(name)
    found = Sys.which(name)
    found === nothing || return found
    for d in _TOOL_DIRS
        cand = joinpath(d, name)
        isfile(cand) && return cand
    end
    return nothing
end

# The cmake to invoke for building generated apps: prefer the absolute path of
# the cmake that built/installed Exasim (recorded at install time) so it works
# from a GUI with a restricted PATH. If the record is missing or its path no
# longer exists (relocated install — recorded cmake is from another machine),
# discover a cmake on this machine; last resort a bare "cmake".
function cmake_command()
    record = joinpath(cmake_dir(), "cmake_command.txt")
    if isfile(record)
        p = strip(read(record, String))
        (!isempty(p) && isfile(p)) && return p
    end
    found = _find_tool("cmake")
    return found === nothing ? "cmake" : found
end
