"""Locate the Exasim install this package belongs to.

The package is installed to <prefix>/lib/pythonX.Y/site-packages/exasim by
`cmake --install`, so the install prefix is normally derived from the package
location itself. Override with the EXASIM_PREFIX environment variable (e.g.
for a source-tree checkout used with `pip install -e`).
"""
import os
import shutil
from pathlib import Path

# Common locations a GUI launchd PATH omits (Homebrew on Apple Silicon / Intel,
# MacPorts, system). Probed by absolute path so discovery does not need PATH.
_TOOL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin")


def _find_tool(name):
    """Locate an executable on this machine: PATH first, then common dirs."""
    found = shutil.which(name)
    if found:
        return found
    for d in _TOOL_DIRS:
        cand = Path(d) / name
        if cand.exists():
            return str(cand)
    return None


def _is_prefix(path):
    return (Path(path) / "lib" / "cmake" / "Exasim" / "ExasimConfig.cmake").is_file()


def install_prefix():
    """Return the Exasim install prefix as a Path, or raise with instructions."""
    env = os.environ.get("EXASIM_PREFIX")
    if env:
        if _is_prefix(env):
            return Path(env)
        raise RuntimeError(
            f"EXASIM_PREFIX={env} does not contain lib/cmake/Exasim/ExasimConfig.cmake")

    # Prefix baked in at install time (EXASIM_PIP_INSTALL writes
    # _installed_prefix.py next to this file).
    try:
        from ._installed_prefix import PREFIX
        if _is_prefix(PREFIX):
            return Path(PREFIX)
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    # Installed layout: <prefix>/lib/pythonX.Y/site-packages/exasim
    for up in (3, 2):
        cand = here.parents[up] if up < len(here.parents) else None
        if cand and _is_prefix(cand):
            return cand
    # Source-tree layout: <repo>/frontends/Python/exasim with the superbuild
    # installed to <repo>-build/install (what tests/run-tests.sh produces),
    # or a local package under <repo>/local.
    if len(here.parents) >= 3:
        repo = here.parents[2]
        for cand in (repo / "local",
                     repo.parent / (repo.name + "-build") / "install",
                     repo / "build" / "install"):
            if _is_prefix(cand):
                return cand
    raise RuntimeError(
        "Cannot locate an Exasim install. Build and install Exasim "
        "(cmake -S <repo> -B <repo>-build && cmake --build <repo>-build && "
        "cmake --install <repo>-build) and/or set EXASIM_PREFIX to the install prefix.")


def cmake_dir():
    return install_prefix() / "lib" / "cmake" / "Exasim"


def frontend_app_template_dir():
    return cmake_dir() / "frontend-app"


def text2code_path():
    return install_prefix() / "bin" / "text2code"


def cmake_command():
    """The cmake executable to invoke for building generated apps.

    Prefer the absolute path of the cmake that built/installed Exasim (recorded
    at install time in lib/cmake/Exasim/cmake_command.txt) so the build works
    from a GUI whose PATH lacks that cmake (e.g. MATLAB launched from the app).
    If that record is missing or its path no longer exists (a relocated install,
    where the recorded cmake belongs to another machine), discover a cmake on
    this machine instead. Last resort: a bare "cmake" (errors clearly if absent).
    """
    record = cmake_dir() / "cmake_command.txt"
    try:
        path = record.read_text().strip()
    except OSError:
        path = ""
    if path and Path(path).exists():
        return path
    return _find_tool("cmake") or "cmake"
