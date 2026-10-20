#!/usr/bin/env bash
# [nonmatching] Phase 9: run a MATLAB non-matching verification command.
#   run-phase9-matlab.sh serialization   phase9_serialization (no install needed)
#   run-phase9-matlab.sh ci              matching 1+1, bypass and the convergence CI subset
#   run-phase9-matlab.sh text2code K     exporttext2code round trip of the coupled
#                                        non-matching example at K+K ranks
# Environment: EXASIM_ROOT, EXASIM_INSTALL. Exit 77 (skip) without MATLAB or an install.
set -uo pipefail
root=${EXASIM_ROOT:?EXASIM_ROOT not set}
if ! command -v matlab >/dev/null 2>&1; then
    echo "SKIP: matlab not found"
    exit 77
fi
cd "$root/tests/nonmatching" || exit 1
case "${1:-}" in
    serialization)
        exec matlab -batch "phase9_serialization" ;;
    ci)
        install=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
        if [[ ! -f $install/lib/cmake/Exasim/ExasimConfig.cmake ]] || ! command -v mpirun >/dev/null 2>&1; then
            echo "SKIP: Exasim install or mpirun unavailable"
            exit 77
        fi
        EXASIM_PHASE9_PREFIX="$install" exec matlab -batch \
            "phase9_fullsolve('matching',1,1); phase9_fullsolve('bypass'); phase9_convergence('ci');" ;;
    text2code)
        install=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
        k=${2:?ranks per model}
        if [[ ! -f $install/lib/cmake/Exasim/ExasimConfig.cmake ]] || [[ ! -x $install/bin/text2code ]] \
            || ! command -v mpirun >/dev/null 2>&1; then
            echo "SKIP: Exasim install, text2code or mpirun unavailable"
            exit 77
        fi
        EXASIM_PHASE9_PREFIX="$install" exec matlab -batch "phase9_text2code($k);" ;;
    phase10-ci)
        install=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
        if [[ ! -f $install/lib/cmake/Exasim/ExasimConfig.cmake ]] || ! command -v mpirun >/dev/null 2>&1; then
            echo "SKIP: Exasim install or mpirun unavailable"
            exit 77
        fi
        EXASIM_PHASE9_PREFIX="$install" exec matlab -batch "phase10_convergence('ci');" ;;
    phase10-partition)
        install=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
        if [[ ! -f $install/lib/cmake/Exasim/ExasimConfig.cmake ]] || ! command -v mpirun >/dev/null 2>&1; then
            echo "SKIP: Exasim install or mpirun unavailable"
            exit 77
        fi
        EXASIM_PHASE9_PREFIX="$install" exec matlab -batch "phase10_partition;" ;;
    phase10-text2code)
        install=${EXASIM_INSTALL:?EXASIM_INSTALL not set}
        k=${2:?ranks per model}
        if [[ ! -f $install/lib/cmake/Exasim/ExasimConfig.cmake ]] || [[ ! -x $install/bin/text2code ]] \
            || ! command -v mpirun >/dev/null 2>&1; then
            echo "SKIP: Exasim install, text2code or mpirun unavailable"
            exit 77
        fi
        EXASIM_PHASE9_PREFIX="$install" exec matlab -batch \
            "phase10_text2code($k,'tri-quad'); phase10_text2code($k,'quad-tri');" ;;
    *)
        echo "usage: run-phase9-matlab.sh serialization|ci|text2code K|phase10-ci|phase10-partition|phase10-text2code K"; exit 1 ;;
esac
