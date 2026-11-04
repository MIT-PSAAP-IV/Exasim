# Exasim codebase-wide ground rules

## Purpose and scope

These rules guide design, implementation, review and verification across the
Exasim backend, frontends, generators, build system, applications, examples and
tests. The priorities are correctness, explicit assumptions, architectural
consistency, simplicity, maintainability, and measured performance.

The central principle is:

> New functionality must extend Exasim's existing infrastructure, not create a
> parallel implementation hidden behind the same user-facing API.

It does not grant implementation approval, replace repository or directory
`AGENTS.md` instructions, or change project-specific authorization. Branch names,
protected files, numerical gates, review responsibilities and phase schedules
remain governed by the active project instructions and explicit maintainer
decisions. Resolve conflicts explicitly rather than silently importing a rule
from another project.

## 1. Design and architectural consistency

1. **Understand before implementing.** Trace at least one representative
   existing example through preprocessing, generation, compilation, library
   linkage, execution and postprocessing. Inspect actual callers and contracts,
   not just similarly named functions.

   Before editing, inspect equivalent Exasim implementations and match their
   data layouts, memory ownership, dispatch, naming and CPU/GPU conventions.
   Reuse established patterns rather than introducing one-off alternatives.
   Explain any necessary deviation and its justification before editing.

2. **Reuse before introducing.** Before adding a function, generator, data
   structure, allocator or execution path, identify the existing infrastructure
   and explain precisely why it cannot satisfy the requirement.

3. **Follow the established execution path.** Native MATLAB, Python and Julia
   examples must use their ordinary `exasim(pde,mesh)` workflows, existing
   generators, standard build machinery and Exasim solver libraries. Text2Code
   must use its established independent route. A matching public function call
   does not establish internal consistency. Verify the actual generated sources,
   provider interface, executable entry point and link commands. Do not replace
   static-solver integration with application-side solver instantiation or
   native generation with a Text2Code wrapper without explicit architectural
   approval.

4. **Keep numerical implementation in the appropriate layer.**

   - Frontends define models, parameters, meshes and controls.
   - Existing generators generate model kernels and required derivatives.
   - Backend code performs numerical integration, assembly, solvers and
     communication.
   - Examples configure and demonstrate functionality; they must not implement
     another solver.

5. **Prefer coherent integration over superficial minimality.** A small patch
   that introduces a bypass, duplicate implementation or hidden dependency is
   not a minimal design. Make localized changes to the correct infrastructure
   instead. Missing infrastructure must be identified, not concealed behind
   another launch path.

6. **Avoid unnecessary frameworks.** Prefer existing structures, ordinary
   functions and explicit data flow. Introduce policies, registries, template
   layers or wrapper classes only when they solve a demonstrated requirement
   and simplify the overall implementation.

## 2. Mathematical correctness and compatibility

7. **Establish the mathematical contract first.** Document governing equations,
   unknowns, units, signs, normals, component ordering, residual definitions and
   required derivative chains before modifying numerical code. Preserve HDG
   local assembly, condensation, trace solves and recovery where applicable.

8. **Differentiate the implemented algorithm completely.** Include dependencies
   introduced by local elimination, gradient recovery, interpolation,
   constitutive integration, temperature-dependent properties and substepping.
   Verify derivatives against independent references. State branch and
   nondifferentiability conventions explicitly.

9. **Preserve existing behavior by default.** New functionality must be
   explicitly selected. Feature-off behavior must preserve established HDG,
   LDG, coupling, adaptivity and time-integration paths. A separate existing-path
   bug fix requires separate justification and evidence. Do not silently change
   precision, ordering, normalization, solver controls or numerical formulas.

10. **Make interface changes explicit.** Document changes to APIs, layouts,
    serialization, provider ABIs, generated kernels and restart formats.
    Version and validate incompatible changes. Never mix incompatible headers,
    generated sources and libraries silently. Explain required rebuilds and
    migration steps.

11. **Preserve physical and numerical state.** Define accepted, stage-base,
    trial and scratch states explicitly. Residual evaluations and rejected
    trials must not corrupt accepted history. Restart must restore all state
    needed to reproduce continuation, not merely primary fields.

## 3. Memory, GPU and MPI

12. **Document storage before using it.** For every numerical workspace,
    identify:

    `owner -> memory space -> capacity -> producer -> consumers -> safe reuse point`

    Check extents, integer conversions and alignment. Raw-pointer views do not
    own storage. Reuse must respect asynchronous kernels, BLAS operations and
    outstanding MPI requests.

13. **Reuse memory safely, not at any cost.** Do not overwrite live residuals,
    matrices, Krylov vectors, histories or rollback data to avoid an allocation.
    If existing capacity is insufficient, document the shortage and propose the
    smallest justified setup-time storage change. Do not introduce another
    memory-owner class or allocator when existing ownership suffices.

14. **Keep hot paths allocation-free.** Avoid numerical allocations, container
    resizing, matrix-inversion workspace allocation and unnecessary host
    transfers during residual, Jacobian, matrix-vector, preconditioner and stage
    evaluations. Allocate genuinely necessary persistent storage during setup,
    under the approved ownership contract.

15. **Make runtime numerical work backend-compatible from the outset.** Use
    existing Kokkos/BLAS conventions, column-major layouts and backend storage.
    Do not introduce host reads of device fields, host-stack kernel arguments,
    managed-memory assumptions or undisclosed CPU fallbacks. One-time
    preprocessing may remain CPU-only. Distinguish small host orchestration
    from numerical field processing.

16. **Treat MPI ownership and synchronization as correctness requirements.**
    Define owned versus ghost data, communicator membership, rank translations,
    payload ordering and tags. Check counts and offsets. Include empty local
    work and nonparticipating ranks where supported. Ensure collective-safe
    failure handling and buffer validity through completion. GPU-aware MPI
    requires producer completion before sends and receive completion before
    device consumption; do not assume cross-library stream ordering.

## 4. User-facing workflows and maintainability

17. **Keep examples concise and directly runnable.** Follow neighboring
    examples' style. Use named physical parameters, explicit units and readable
    setup. Avoid mandatory environment-variable machinery, private test
    harnesses, cached executables or fixed `/tmp` dependencies. Document actual
    launch commands, dependencies, output locations and supported options.

18. **Keep frontend physics consistent.** MATLAB, Python, Julia and Text2Code
    routes must represent the same mathematical model, parameters and solver
    controls for delivered configurations. Share backend algorithms; do not
    implement separate numerical physics in each language. Record unsupported
    or untested routes honestly rather than providing export-only stubs as
    runnable examples.

19. **Use standard artifact locations.** Generated application sources and
    build products belong under the ordinary `.exasim` build tree. Preserve
    established input/output conventions. Disposable verification copies, large
    results and profiling artifacts belong in dedicated `/tmp` directories;
    archive required evidence durably. Do not modify an installed prefix or
    setup script as an incidental implementation step.

20. **Keep source changes readable and focused.** Follow existing naming,
    layout and ownership conventions. Avoid unrelated formatting, speculative
    refactoring, duplicated helpers and dead compatibility code. Comments
    should explain mathematical meaning, ownership or a non-obvious constraint.
    Keep essential capability guards and failure checks; put exhaustive
    independent oracles in tests rather than repeating them in hot paths.

## 5. Verification and performance

21. **Define acceptance before measuring the candidate.** Specify metrics,
    scaling, tolerances, reference solutions and supported configurations.
    Do not weaken a gate, suppress warnings or change settings merely to obtain
    a passing result. Maintainer-approved gate amendments must preserve the
    original results and distinguish policy changes from numerical fixes.

22. **Verify from small independent tests to complete workflows.** As
    applicable, require:

    - Analytic or manufactured solutions.
    - Local residual and derivative checks.
    - Complete assembled-Jacobian checks.
    - Conservation, admissibility and spatial/temporal convergence checks.
    - Repeated evaluation, rejection and restart checks.
    - Feature-off regression preservation.
    - Frontend parity and MPI partition comparisons.
    - Memory-safety checks and targeted negative/mutation tests.

    Use targeted tests first. Run expensive suites only within the authorized
    scope and follow the active project's implementer/reviewer test assignment.
    Keep routine CI tests bounded; provide explicit commands for longer manual
    checks. Never compute convergence rates from unconverged solutions.

23. **Prove performance claims.** Compare equivalent inputs, compiler flags,
    backends, tolerances and convergence outcomes. Measure setup cost, runtime,
    allocations, memory, transfers, synchronization and communication where
    relevant. Consider kernel launch counts, register pressure, coalescing and
    load balance. Report regressions and inconclusive measurements honestly.
    Do not compare optimized and unoptimized builds as algorithmic speedups.

24. **Separate portability from validation.** CPU success does not establish
    threading, MPI, CUDA, HIP or GPU-aware MPI correctness. Compilation is not
    hardware validation. Record each configuration separately as `PASS`,
    `FAIL`, `NOT RUN`, `BLOCKED` or `UNSUPPORTED`, with explicit scope.

## 6. Change control and completion

25. **Record provenance and preserve user work.** Before editing, record the
    branch, commit and working-tree state. Identify affected files, assumptions,
    compatibility implications and verification commands. Do not discard
    unrelated changes or perform Git writes, installation or publication without
    authorization. Follow `docs/install/ai-agents.md` for installation-related
    tasks.

26. **Require evidence for completion.** Implemented, tested, ready for review
    and approved are different states. Completion requires the specified
    workflow to work through the intended architecture, not a substitute
    fixture or bypass. Preserve unresolved failures and historical evidence.
    Reports must include actual commands, source and generated-code provenance,
    compiler/backend configuration, controls, results and remaining limitations.
    Never manufacture results or describe unexecuted checks as passing.

    Before committing, explicitly review consistency with the equivalent Exasim
    implementations, not just whether tests pass. Resolve unintended deviations
    or document necessary ones and their compatibility implications.

## Mandatory design gate

Before implementation, every substantial change must answer:

1. What existing path does this extend?
2. What existing code and memory does it reuse?
3. Why is each new component necessary?
4. What remains unchanged when the feature is disabled?
5. How are complete numerical dependencies preserved?
6. How will correctness, ordinary launch-path consistency and performance be
   demonstrated?

If these answers are unclear, investigate before adding code. If the approved
requirements cannot be satisfied safely, report the exact conflict and evidence
instead of silently changing the architecture or mathematics.

