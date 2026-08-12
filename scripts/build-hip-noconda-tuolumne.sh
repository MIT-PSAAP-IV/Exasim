#!/usr/bin/env bash
# Build Exasim MPI+GPU (HIP) on tuolumne WITHOUT conda — system toolchain only.
# Runs ON the remote (shipped by submit-tuo.sh), not locally.
#
#     bash tuo-build-exasim-noconda.sh <pass> [remote-root]
#
# Uses system gcc 13.3.1, Cray MPICH, ROCm, OpenBLAS, and cmake.
# Exasim vendors Kokkos, SymEngine, text2code, METIS/ParMETIS.
# No Python frontends (system python is 3.6.8). No PETSc.
#
# Installs to passes/<pass>/install/Exasim-hip-noconda so it sits alongside
# the conda-based Exasim-hip without interfering.
set -euo pipefail

PASS="${1:-}"; P4_REMOTE_ROOT="${2:-/p/lustre5/collin1/psaap4p-tuo}"
[[ -n "${PASS}" ]] || { echo "usage: tuo-build-exasim-noconda.sh <pass> [remote-root]" >&2; exit 1; }

MODE="hip-noconda"
PASS_ROOT="${P4_REMOTE_ROOT}/passes/${PASS}"
P4_SRC="${PASS_ROOT}/src/Exasim"
P4_BUILD="${PASS_ROOT}/build/Exasim-${MODE}"
P4_INSTALL="${PASS_ROOT}/install/Exasim-${MODE}"
P4_LOG="${PASS_ROOT}/log"
mkdir -p "${P4_BUILD}" "${P4_INSTALL}" "${P4_LOG}"
[[ -d "${P4_SRC}" ]] || { echo "no source at ${P4_SRC} — run etc/remote/sync.sh first" >&2; exit 1; }

export TMPDIR="${P4_REMOTE_ROOT}/tmp"
mkdir -p "${TMPDIR}"

# --- toolchain (all system, no conda) ------------------------------------
P4_ROCM_VERSION="${P4_ROCM_VERSION:-6.3.1}"
P4_ROCM="/opt/rocm-${P4_ROCM_VERSION}"
P4_GCC_VERSION="${P4_GCC_VERSION:-13.3.1}"
P4_GCC="/usr/tce/packages/gcc/gcc-${P4_GCC_VERSION}"

[[ -d "${P4_GCC}" ]] || { echo "FATAL: gcc not found at ${P4_GCC}" >&2; exit 1; }
[[ -d "${P4_ROCM}" ]] || { echo "FATAL: ROCm not found at ${P4_ROCM}" >&2; exit 1; }

P4_MPICH=""
if [[ -d /usr/tce/packages/cray-mpich-tce ]]; then
    P4_MPICH="$(ls -d /usr/tce/packages/cray-mpich-tce/cray-mpich-*-gcc-"${P4_GCC_VERSION}" 2>/dev/null \
        | sort -V | tail -1 || true)"
fi
[[ -n "${P4_MPICH}" && -d "${P4_MPICH}" ]] || { echo "FATAL: Cray MPICH for gcc-${P4_GCC_VERSION} not found" >&2; exit 1; }

export PATH="${P4_ROCM}/bin:${P4_MPICH}/bin:${P4_GCC}/bin:/usr/bin:/usr/sbin:/bin:/sbin"
export LD_LIBRARY_PATH="${P4_ROCM}/lib:${P4_MPICH}/lib:${P4_GCC}/lib64"
export CRAY_MPICH_DIR="${P4_MPICH}"

export CC="${P4_GCC}/bin/gcc"
export CXX="${P4_GCC}/bin/g++"
export FC="${P4_GCC}/bin/gfortran"

echo "=== no-conda Exasim HIP build ==="
echo "==> host      $(hostname)"
echo "==> pass      ${PASS}"
echo "==> src       ${P4_SRC}"
echo "==> cmake     $(cmake --version | head -1)"
echo "==> gcc       $(${CC} --version | head -1)"
echo "==> mpi       ${P4_MPICH}"
echo "==> rocm      ${P4_ROCM}"
echo "==> hipcc     $(hipcc --version 2>&1 | head -1)"
echo "==> blas      /usr/lib64/libopenblas.so"
echo "==> ccache    $(ccache --version 2>/dev/null | head -1 || echo 'not found')"
echo ""

# GCC 13.3.1 backports -Wtemplate-body from GCC 14; vendored SymEngine
# headers trigger it. Do NOT export CXXFLAGS — it leaks into the vendored
# Kokkos build, whose hipcc rejects the gcc-only flag. Pass it only via
# CMAKE_CXX_FLAGS (which the superbuild forwards to the inner solver build
# but NOT to vendored deps like Kokkos). text2code's runtime compilation
# inherits CXXFLAGS, so set it only in the inner build's environment later.

ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${P4_INSTALL}"
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_C_COMPILER="${CC}"
    -DCMAKE_CXX_COMPILER="${CXX}"
    -DCMAKE_CXX_FLAGS="-Wno-template-body"

    -DEXASIM_HIP=ON
    -DEXASIM_CUDA=OFF
    -DEXASIM_MPI=ON
    -DEXASIM_NOMPI=ON
    -DEXASIM_GPU_ARCH=gfx942
    -DEXASIM_BUILD_TESTS=ON

    -DMPI_C_COMPILER="${P4_MPICH}/bin/mpicc"
    -DMPI_CXX_COMPILER="${P4_MPICH}/bin/mpicxx"
    -DMPI_HOME="${P4_MPICH}"

    -DWITH_PARMETIS=ON
    -DEXASIM_FRONTENDS=OFF

    -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so
    -DLAPACK_LIBRARIES=/usr/lib64/liblapack.so
)

JOBS="${JOBS:-16}"

echo "==> configure -> ${P4_BUILD}"
cmake -S "${P4_SRC}" -B "${P4_BUILD}" "${ARGS[@]}" 2>&1 | tee "${P4_LOG}/exasim-${MODE}-configure.log"

echo ""
echo "==> build -j${JOBS}"
cmake --build "${P4_BUILD}" -j"${JOBS}" 2>&1 | tee "${P4_LOG}/exasim-${MODE}-build.log" | tail -20

echo ""
echo "==> install -> ${P4_INSTALL}"
cmake --install "${P4_BUILD}" > "${P4_LOG}/exasim-${MODE}-install.log" 2>&1

cat > "${P4_INSTALL}/.psaap4p-pass" <<EOF
pass=${PASS}
component=Exasim-${MODE}
pass_root=${PASS_ROOT}
host=$(hostname)
built=$(date -u +%Y-%m-%dT%H:%M:%SZ)
toolchain=system-only (no conda)
gcc=${P4_GCC_VERSION}
mpi=cray-mpich $(basename "${P4_MPICH}")
rocm=${P4_ROCM_VERSION}
blas=system-openblas
EOF

if [[ -d "${P4_INSTALL}/lib64" ]]; then
    echo "==> merging lib64/ -> lib/"
    mkdir -p "${P4_INSTALL}/lib"
    cp -a "${P4_INSTALL}/lib64/"* "${P4_INSTALL}/lib/" 2>/dev/null || true
    rm -rf "${P4_INSTALL}/lib64"
    for _f in "${P4_INSTALL}"/lib/cmake/Exasim/*.cmake; do
        [[ -f "${_f}" ]] && sed -i 's|/lib64/|/lib/|g' "${_f}"
    done
fi

echo ""
echo "==> DONE (no-conda HIP build)"
echo "==> installed libs:"
ls "${P4_INSTALL}/lib/"*.a 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
