# Extending Exasim With AI

AI tools can help draft backend, frontend, Text2Code, application, and
documentation changes. For core solver work, use AI as an assistant, not as an
authority. Numerical correctness and Exasim architecture must drive the design.

## Suitable Extension Tasks

- Add a frontend option that serializes into existing backend fields.
- Add a documentation page or example.
- Add a new Text2Code keyword that maps to an existing runtime field.
- Add a small utility function with tests.
- Improve error messages.
- Create a minimal app demonstrating an existing capability.

## High-Risk Extension Tasks

Be more cautious with:

- new discretizations;
- new preconditioners;
- GPU kernels;
- MPI communication paths;
- file-format changes;
- changes to `commonstruct`, `appstruct`, or solution memory ownership;
- nonlinear solver or HDG static-condensation modifications.

These require explicit design review, small tests, and numerical validation.

## Backend Extension Prompt

```text
Create a design plan before editing.

Task:
- add <feature> to Exasim backend

Affected areas:
- backend/...
- frontends/...
- text2code/...

Requirements:
- preserve existing interfaces
- maintain HDG static-condensation structure
- avoid unnecessary host-device copies
- support CPU and GPU or explain why not

Before coding:
- identify data ownership
- identify array sizes and layout
- list tests and numerical checks
```

## Frontend Extension Prompt

```text
Add a new frontend option named <field>.

Requirements:
- initialize it in MATLAB/Python/Julia defaults
- serialize it consistently
- preserve existing examples
- add one minimal example or test
- update docs/reference/pdeapp.md if it reaches text input

Keep changes minimal and list affected files before editing.
```

## GPU Optimization Prompt

```text
Analyze this Exasim GPU kernel path using the profiler output below.
Do not rewrite the kernel yet.

Profiler evidence:
- kernel timings:
- memory throughput:
- occupancy:
- launch counts:

Identify whether the bottleneck is memory bandwidth, launch overhead,
uncoalesced access, register pressure, or host-device synchronization.
Propose one minimal experiment to validate the diagnosis.
```

## Documentation Extension Prompt

```text
Update Exasim docs for <feature>.
Use current source code as the authority.
Do not document unsupported options.
Add cross-links to Quickstart, Frontends, Physics Models, Theory, and Reference.
Run mkdocs build --strict.
```
