# PLAN — precision threading (branch teoc-properly-separate-out)

Goal: thread scalar precision `T` (default `dstype`) + index `I` (default `Int`) through the Exasim
C++ backend so precision is a type-level choice, byte-identical under defaults. Frontends
(Matlab/Python/Julia) stay UNCHANGED — the cut is the ExasimDriverABI fn-pointer seam (stance A:
static_assert(T==dstype) in EXASIM_DRIVER_CALL AbiAdapter branch; forward-activating).

## Done + pushed
- P0 boundary (floatTy/intTy + PETSc ABI guard): 410a8abb
- P1 structs (10 core): 5dee590a, 4b23998b
- P2a CDiscretization<T,I> (extern template + explicit instantiation, TU split): 53e99346
- P2b CResidual/CAssembler/CPreconditioner/CSolver <M,T,I>: fa1f84a4
- P2c PETSc shim Operator/ShellMat<Scalar,Idx> + per-inst ABI assert: fdadcc6c
- P3 blas<T> trait + CPU GEMM/GETRF wrappers (pblas.h): 338bb652
- P3 cut stance A (EXASIM_DRIVER_CALL static_assert): 4e15bc5f
- P3 mpi_type<T> trait + fix hardcoded MPI_DOUBLE halo-exchange bug (34 sites): 6f8013df
- P3 pblas transpose GEMM/GEMV via blas<T>: ac4ebf8b
- P3 cpuimpl.h 8 CPU primitives <T=dstype>: 5973ac67
- P3 kokkosimpl.h all 137 Kokkos device kernels <Ty=dstype>: 4f227870 (GPU-verified aed7402d)
- P3 Discretization/*.hpp HDG/LDG free-function layer <M,T=dstype,I=Int> (uequation/matvec/qequation/
  wequation/residual/qoicalculation/massinv/getuhat/setstructs): 304d4a28 (GPU-verified c10f18fd)
- P3 pblas tail (Inverse/PDOT/PNORM/DOT/Array*/NORM/PGEMNMStridedBached + blas<T> level-1 ops): 3461759a
  (GPU-verified this commit)

## Phase 3 status: COMPUTE + ORCHESTRATION COMPLETE
The whole solve path is now templated + byte-identical on CPU/CPU-MPI/GPU/GPU-MPI: structs, FEM classes,
compute primitives (pblas/cpuimpl/kokkosimpl), the Discretization HDG/LDG free-function layer, and the
PETSc shim. ONLY remaining P3 tail: the blas<T> GPU trait methods (cublas/hipblas) for *non-default* GPU
precision (default-dstype GPU path already verified). Then Phase 4 (codegen) + Phase 5 (mixed-prec test).

## Verified
- CPU + CPU-MPI (EXASIM_MPI=ON local): build_robust ALL PASS + prec_fullverify app-regression at
  unchanged rel_L2 (poisson3d 9.99e-11, isoq3d 1.95e-9, poisson2d 2.29e-12) every increment.
- GPU (dgx-b V100): @4e15bc5f CUDA build compiles nvcc/Kokkos-CUDA; petsc_poisson EXASIM_BACKEND=2
  PASS (ShellMat vs op.mat()=0, MATAIJ 2.6e-16). GPU-MPI (petsc_poisson_mpi) verification IN PROGRESS
  for the mpi_type commit (6f8013df).

## Verify loops
- Local CPU/CPU-MPI: /tmp/build_robust.sh (fast; sync edited headers to Exasim-build/install/include
  first via a BASH loop — zsh won't word-split), then /tmp/prec_fullverify.sh (full rebuild + MPI +
  tests/run-app-regression.sh). MPI code is #ifdef HAVE_MPI; local build has it (np=4 tests run).
- Remote GPU: bash ~/projects/psaap4/remote/sync.sh dgx (git archive HEAD -> dgx-b
  /data/scratch/teoc/exasim-teoc; COMMIT FIRST). Then ssh dgx-b bash /data/scratch/teoc/build-dgx.sh
  (CUDA gpu+gpumpi install, ~15min). petsc_poisson GPU: build-petsc-gpu-consumer.sh + EXASIM_BACKEND=2.
  petsc_poisson_mpi GPU-MPI: build-run-petsc-gpumpi.sh (mpiexec -n 1/2/4, backend=2).
  NB: GPU build + EXASIM_BACKEND=0 is an UNSUPPORTED combo (Kokkos is CUDA-space) -> cpuComputeInverse
  fails; not a regression. Supported: CPU build->backend 0, GPU build->backend 2.

## Established pattern (shadow-alias, byte-identical under defaults)
- structs/classes: template<class T=::dstype,class I=::Int> + member `using dstype=T; using Int=I;`
  (+ per-struct `using solstruct=solstructT<T,I>;`), body unchanged, `using X=XT<::dstype,::Int>;`.
- traits: blas<T> (pblas.h), mpi_type<T> (common.h) replace #ifdef USE_FLOAT / hardcoded types.
- CDiscretization only: needs extern template (discretization.h) + explicit instantiation
  (discretization.cpp) because main.cpp builds CSolution<M> holding it by value across the TU split.

## Remaining Phase 3 (large, GPU-heavy)
- pblas.h: rest of wrappers (Gauss2Node1, PGEMTM/PGEMNV/PGEMTV, PGEMNMStridedBached, PDOT/DOT,
  Inverse) + GPU trait methods (cublas/hipblas) — GPU verified on dgx-b.
- Thread `using dstype=T` into comm helpers so mpi_type<dstype>() auto-becomes mpi_type<T>().
- Discretization/*.hpp kernels take T* (uequation 156, wequation 55, qequation 43, matvec, massinv).
- Common/cpuimpl.h (37), Common/kokkosimpl.h (880 — Kokkos::View<dstype*> -> View<T*>).

## Deferred
- Seam upgrade B (T<->dstype conversion shim) or Phase 4 (template codegen) — only if
  frontend-DEFINED PDEs need mixed precision. A locks in nothing (localized to the one macro branch).
- Phase 5 cutover + a real single-precision-inner-solve test.
