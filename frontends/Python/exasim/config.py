"""Locate the Exasim install this package belongs to.

The package is installed to <prefix>/lib/pythonX.Y/site-packages/exasim by
`cmake --install`, so the install prefix is normally derived from the package
location itself. Override with the EXASIM_PREFIX environment variable (e.g.
for a source-tree checkout used with `pip install -e`).
"""
import os
from pathlib import Path


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
    # installed to <repo>-build/install (what tests/run-tests.sh produces).
    if len(here.parents) >= 3:
        repo = here.parents[2]
        for cand in (repo.parent / (repo.name + "-build") / "install",
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
