# Debugging And Code Review With AI

AI tools can help diagnose failures, but they need concrete evidence. Provide
the exact error, command, relevant files, and expected behavior.

## Build Errors

Good prompt:

```text
I get this build error when compiling an Exasim app:

<paste full compiler/linker output>

Command:
cmake -S . -B build -DExasim_DIR=$EXASIM_PREFIX
cmake --build build

Relevant files:
- CMakeLists.txt
- exasimapp.cpp
- pdeapp.txt

Identify whether this is a missing include, wrong Exasim target, stale build
cache, or package-location issue. Propose the minimal fix.
```

## Runtime Errors

For runtime failures, include:

- command line;
- MPI rank count;
- CPU/GPU backend;
- `pdeapp` or frontend driver;
- first error message, not only the final abort;
- relevant `dataout` or residual files if available.

## Convergence Problems

AI can help reason about convergence if you provide:

- residual history;
- Newton and GMRES settings;
- time step size;
- mesh resolution and polynomial order;
- boundary conditions;
- physical parameters;
- whether the same case works at lower order or lower Reynolds number.

Prompt:

```text
This Exasim Navier-Stokes case stalls after Newton iteration 2. Here are the
residual logs, pdeapp settings, boundary IDs, and physicsparam values. Diagnose
whether the likely cause is boundary conditions, time step size, mesh quality,
preconditioning, or invalid physical states. Do not change code yet; propose
diagnostic runs first.
```

## Code Review Prompts

Use AI as a reviewer:

```text
Review these Exasim changes for correctness. Focus on:
- array sizes and column-major layout
- host/device memory consistency
- MPI rank-local file names
- conservation and boundary-condition consistency
- backward compatibility

Report findings by severity with file and line references.
```

## What Not To Do

- Do not paste only the last line of an error.
- Do not ask the AI to "fix convergence" without physical and numerical
  context.
- Do not accept a large refactor when a stale CMake cache or wrong
  `Exasim_DIR` is the likely issue.
- Do not use AI-generated fixes on production runs before reproducing the bug
  on a small case.
