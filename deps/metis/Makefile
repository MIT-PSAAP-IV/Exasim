# ---- CPU core detection (Linux/macOS/WSL/MSYS2) ----
NPROCS := $(strip $(shell \
  (command -v nproc >/dev/null 2>&1 && nproc) || \
  (command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu 2>/dev/null) || \
  (command -v getconf >/dev/null 2>&1 && getconf _NPROCESSORS_ONLN 2>/dev/null) || \
  echo 1))

SHELL := /usr/bin/env sh

# Keep the sub-build environment clean without dropping variables required by
# CMake, Homebrew, and macOS compiler discovery.
CLEAN_ENV = env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" USER="$$USER" TMPDIR="$$TMPDIR" SDKROOT="$$SDKROOT" MACOSX_DEPLOYMENT_TARGET="$$MACOSX_DEPLOYMENT_TARGET"
# ParMETIS needs MPI compiler-wrapper state from the loaded environment on HPC
# systems, so do not scrub the environment for that sub-build.
PRESERVE_ENV = env MAKEFLAGS= MFLAGS= GNUMAKEFLAGS=

# Optional C compiler forwarded by the superbuild, e.g. cc=amdclang on ROCm
# systems where the Cray cc wrapper requires a PrgEnv module.
cc ?= not-set
parmetis_cc ?= $(cc)
CONFIG_CC =
ifneq ($(strip $(cc)),not-set)
ifneq ($(strip $(cc)),)
CONFIG_CC = cc="$(cc)"
endif
endif
PARMETIS_CONFIG_CC =
ifneq ($(strip $(parmetis_cc)),not-set)
ifneq ($(strip $(parmetis_cc)),)
PARMETIS_CONFIG_CC = cc="$(parmetis_cc)"
endif
endif

# ---- stop leaking parent make flags into sub-makes ----
MAKEFLAGS :=
MFLAGS :=
GNUMAKEFLAGS :=
unexport MAKEFLAGS
unexport MFLAGS
unexport GNUMAKEFLAGS

.PHONY: metis

metis:
	@echo "Building with -j$(NPROCS)…"

	# GKlib (plain make, but still keep env clean for safety)
	cd GKlib && \
	  $(CLEAN_ENV) make config prefix="$$PWD" $(CONFIG_CC) && \
	  $(CLEAN_ENV) make install

	# METIS: requires 'make config'; run in a clean environment to avoid the 'w' target
	cd METIS && \
	  $(CLEAN_ENV) make config prefix="$$PWD" gklib_path="$$PWD/../GKlib" $(CONFIG_CC) && \
	  $(CLEAN_ENV) make -j"$(NPROCS)" install

  # ParMETIS: 
	cd ParMETIS && \
	  $(PRESERVE_ENV) make config prefix="$$PWD" gklib_path="$$PWD/../GKlib" metis_path="$$PWD/../METIS" $(PARMETIS_CONFIG_CC) && \
	  $(PRESERVE_ENV) make -j"$(NPROCS)" install

	# preprocess (TEST_RUN_AFTER_BUILD=OFF: build the tool, don't run a sample model)
	cd preprocessing && cmake -S . -B build -DTEST_RUN_AFTER_BUILD=OFF && cmake --build build -j"$(NPROCS)"
