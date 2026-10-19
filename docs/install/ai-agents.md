# AI-Assisted Installation

Use this page when asking an AI coding agent to download, configure, build,
install, or verify Exasim.

> **Prompt for an AI agent:** Install Exasim on this machine by following
> <https://mit-psaap-iv.github.io/Exasim/install/ai-agents/>. Inspect first,
> propose an explicit plan, wait for my approval, and verify the installation.

This page adds agent policy and routing to the maintained installation manual.
It is not a second installation manual. The agent must use the current source,
the relevant pages under `docs/install/`, and the checked-out CMake files as the
technical authorities.

## Step 0: No Repository Yet

Before cloning, the agent must:

1. Confirm the source repository and requested revision. Unless the user says
   otherwise, propose `https://github.com/MIT-PSAAP-IV/Exasim.git` and its
   current default branch.
2. Choose a parent directory whose path contains no whitespace and clone into a
   child directory named exactly `Exasim`.
3. Show the repository URL, revision, destination, and network operation, then
   obtain approval before downloading or writing the source.
4. After cloning, report the full commit hash and re-read the local
   `docs/install/ai-agents.md`. If it differs materially from the published
   page, use the local page for that revision and explain the difference.

If the target machine is offline, request a user-provided Git bundle or source
archive instead of attempting a network operation. Verify its origin and
revision as far as the available evidence permits.

## Governing Rules

- Inspect before acting.
- Use an out-of-source build directory that is also outside the source tree.
- Require an explicit, user-writable `CMAKE_INSTALL_PREFIX`; never accept
  CMake's `/usr/local` default silently.
- Explain CPU, MPI, CUDA, HIP, frontend, resource, and verification choices.
- Obtain approval before installing packages, loading persistent modules,
  downloading optional dependencies, writing outside approved directories,
  using elevated privileges, or launching expensive jobs.
- Never use `sudo` unless the user explicitly requests and approves it.
- Never weaken tests, solver tolerances, or application inputs to make a build
  appear successful.
- Preserve local source changes and existing installations.
- Keep complete configure, build, install, and verification logs.

The response-format, code-change, and solver-modification rules in the root
`AGENTS.md` do not govern installation work. Its correctness, explicit-
assumption, approval, minimal-change, and verification principles still apply.

## Do Not Use Legacy Installers

Do not use `install/install.py`, `install/install.m`, `install/install.jl`,
`install/*.txt`, or `install/*.docx` as installation authority. Some are legacy
package installers or stale platform notes. Use this page, the maintained
installation pages, and current CMake source instead.

## Inspect the Machine Read-Only

Before proposing commands, record what is relevant to the request:

- operating system, version, and CPU architecture;
- available memory, disk space, and reasonable build parallelism;
- Git and CMake paths and versions;
- C and C++ compiler paths and versions;
- MPI wrappers, launcher, and implementation when MPI is requested;
- NVIDIA driver, visible GPU, CUDA compiler, and architecture for CUDA;
- AMD GPU, ROCm tools, HIP compiler, and architecture for HIP;
- MATLAB, Python, and Julia availability when requested;
- loaded HPC modules and site restrictions;
- existing source, build, and installation directories.

Executable presence is not proof of usability. Report compiler/MPI or
driver/toolkit inconsistencies and unresolved uncertainty. During inspection,
do not install packages, edit shell startup files, persist module changes, or
modify the source tree.

## Read the Authoritative Sources

Read in this order:

1. [Installation decision tree](index.md).
2. The one platform page selected by that decision tree.
3. The root `CMakeLists.txt` and `install/CMakeLists.txt` for supported options.
4. [CMake reference](../reference/cmake.md) for option meaning.
5. [Verification and testing](verification.md) for post-install checks.

Do not copy a remembered option list. Discover current options from `option(`
and cache-variable declarations in both CMake files. After configuration, run
`cmake -LH <build-directory>` and record the effective values. If code and docs
disagree, the checked-out code defines what exists; report the documentation
gap instead of guessing.

For a superbuild, only options defined in the root `CMakeLists.txt` and explicitly
forwarded through `ExternalProject_Add(exasim_build ...)` can affect the solver
sub-build. A solver-layer-only option passed to the superbuild may be ignored.
Treat CMake's `Manually-specified variables were not used` warning as a stop
condition. Record effective solver-layer values from the `exasim_build` sub-build
cache as well as the top-level output of `cmake -LH <build-directory>`.

## Propose the Plan and Wait

Before configuration, present:

- source repository, revision, and source directory;
- out-of-source build directory;
- explicit installation prefix; when the user has no preference, propose a
  writable sibling directory such as `<source-parent>/exasim_install`;
- compilers and MPI implementation;
- enabled and disabled Exasim options and why;
- at most one GPU backend and its detected architecture;
- requested frontend installation behavior;
- when Python or Julia frontends are requested, use
  `-DCMAKE_INSTALL_LIBDIR=lib`; their current prefix discovery does not find an
  Exasim CMake package installed only under `lib64/cmake/Exasim`;
- any option that may access the network or mutate Python or Julia environments;
- build parallelism, MPI ranks, GPU use, wall-time expectations, and disk use;
- focused verification commands and their success criteria.

Wait for explicit approval. A failure that requires changing the compiler,
backend, prefix, dependency source, or verification scope requires an amended
plan and renewed approval.

## Configure, Build, and Install

Use the approved values with the standard pattern:

```bash
cmake -S <source> -B <build> \
  -DCMAKE_INSTALL_PREFIX=<prefix> \
  <approved-options>
cmake --build <build> --parallel <approved-count>
cmake --install <build>
```

Show exact commands before running them. Do not silently select another
compiler, MPI stack, GPU architecture, branch, revision, or installation prefix.
Do not enable CUDA and HIP in the same build.

The default CPU superbuild uses vendored source after the repository is present.
Optional features may still access the network or mutate language environments;
identify these from the current CMake source and include them in the approved
plan. For an offline installation, leave network-dependent options disabled.

## HPC and GPU Systems

On an HPC system:

- inspect `module list` and relevant `module avail` output;
- include session-level module changes in the approved plan; do not persist them
  or edit startup files;
- obey login-node and scheduler policies;
- if builds or tests require a compute node, prepare commands or a batch script
  and let the user approve or submit it;
- include the resolved module list in the final report.

Use the site's approved launcher, such as `srun`, `flux run`, or `mpirun`, inside
an allocation when required. Never launch MPI or GPU smoke tests on a login node
when site policy requires a compute allocation.

For CUDA or HIP, compilation is not runtime validation. If no appropriate GPU
node is available, report the backend as **built, not executed**. Verify a GPU
backend only after a representative executable runs successfully on that GPU.

## Focused Verification

Unless the user approves a broader scope, perform focused verification:

1. Confirm expected installed headers, libraries, executables, runtime data,
   frontends, and exported CMake package files.
2. Verify `find_package(Exasim)` from a copied or temporary consumer directory
   outside the source tree.
3. Run one small CPU example.
4. Run a small MPI smoke test when MPI was built.
5. Run a small GPU smoke test only when a compatible GPU is available.
6. Exercise each requested frontend sufficiently to establish that it finds the
   installed prefix.

Build or run the CTest suite only when its scope and resource cost were approved.
Use named tests or filters for focused verification. Run Text2Code and examples
from copied or temporary case directories so generated files do not modify the
source checkout.

Distinguish **passed**, **failed**, **skipped**, **unavailable**, and
**untested**. A successful compilation alone is not a verified installation.

## Stop Conditions

Stop and report evidence when:

- the requested revision cannot be obtained;
- the proposed destination already exists and is non-empty or has unpreserved
  changes;
- no suitable whitespace-free source, build, and installation paths are
  available;
- available disk space is below the plan's estimate;
- a required tool is missing or incompatible;
- the requested backend is unsupported on the detected hardware;
- an offline host is asked to enable an option that requires network access;
- CMake reports that a manually specified variable was not used;
- the selected prefix would overwrite an installation without approval;
- configuration, compilation, installation, or required verification fails;
- continuing requires a plan change or prohibited system modification.

Do not report a timeout as solver non-convergence, or solver non-convergence as
an installation failure, without supporting evidence.

## Final Report

Report:

- repository URL, branch or tag, and full commit hash;
- source, build, and installation paths;
- operating system and relevant compiler, CMake, MPI, GPU, module, and frontend
  versions;
- effective CMake cache options from both the top-level superbuild and the
  solver `exasim_build` sub-build;
- exact configure, build, install, and verification commands;
- installed artifacts and package-discovery result;
- tests and examples actually run, with observed outcomes;
- enabled, unavailable, skipped, and untested capabilities;
- approved environment actions still required from the user, including the
  value of `EXASIM_PREFIX` when frontends or consumers need it;
- warnings, limitations, and recommended follow-up work.

Installation is complete only when installed artifacts are discoverable and the
approved smoke verification succeeds.
