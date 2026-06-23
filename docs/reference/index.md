# Reference

The Reference section is the precise API and configuration reference for
Exasim. It complements the tutorials and conceptual guides by documenting
exact interfaces, accepted input forms, runtime file semantics, model
functions, and build-system options.

Use this section when you need to answer questions such as:

- Which `pdeapp.txt` keys are recognized?
- Which frontend `pde` fields control a runtime option?
- Which model functions are required by Text2Code?
- What arguments does a boundary or source function receive?
- Which CMake targets and options are installed?
- How are parameter sweeps and postprocessing configured exactly?

## Relationship to Other Documentation

| Section | Use it for |
| --- | --- |
| [Quickstart](../getting-started/quickstart.md) | First successful installation and simulation. |
| [Application Modes](../usage-modes/index.md) | User workflows such as built-in, shared-library, sweep, and postprocess modes. |
| [Using AI Tools](../using-ai-tools/index.md) | Prompting AI assistants to generate examples or modify code safely. |
| [Physics Models](../physics-models/index.md) | Mathematical meaning of ModelC, ModelD, ModelW, EOS, AV, and coupling. |
| [Theory](../theory/index.md) | DG/HDG/LDG, solvers, preconditioning, and numerical algorithms. |
| [Internals](../internals/index.md) | Implementation architecture and contributor workflow. |
| Reference | Exact fields, functions, signatures, file formats, and build options. |

## Reference Map

| Page | Contents |
| --- | --- |
| [pdeapp reference](pdeapp.md) | `pdeapp.txt` keys, frontend `pde` fields, defaults, accepted values, interactions. |
| [pdemodel reference](pdemodel.md) | `pdemodel.txt` declarations, function blocks, generated outputs. |
| [Model contract](model-contract.md) | C++ model/provider contract and data layout conventions. |
| [Model functions](model-functions.md) | Function-by-function interface reference for fluxes, sources, boundaries, initial conditions, visualization, and QoI. |
| [Equations of state](equations-of-state.md) | EOS callbacks and derivative interfaces. |
| [Boundary conditions](boundary-conditions.md) | Boundary IDs, expressions, `Ubou`, `Fbou`, and HDG boundary residuals. |
| [Source terms](source-terms.md) | Volume source and auxiliary source interfaces. |
| [Auxiliary equations](auxiliary-equations.md) | `w` equations, `Sourcew`, initialization, and coupling to flux/source/EOS. |
| [Parameter sweeps](parameter-sweeps.md) | Frontend and standalone sweep configuration, warm-start, metadata, outputs. |
| [Postprocessing](postprocessing.md) | Postprocessing controls, saved solution replay, visualization and QoI outputs. |
| [Text2Code](text2code.md) | Text2Code executable behavior, parser format, output files, and examples. |
| [CMake](cmake.md) | Superbuild options, install options, exported targets, components, examples. |
| [Glossary](glossary.md) | Common Exasim terms and abbreviations. |

## Required Versus Optional

This Reference uses the following conventions:

- **Required** means the parser or runtime needs the field for the workflow.
- **Optional** means Exasim has a default or the feature is inactive when the
  field is omitted.
- **Frontend-only** means the setting exists in MATLAB/Python/Julia helper
  logic but is not parsed from `pdeapp.txt` or stored in `app.bin`.
- **Runtime** means the setting is serialized into Exasim binary input files
  and affects standalone C++ execution.

## AI-Assisted Development

AI tools should treat this Reference as the source of truth for exact names,
interfaces, and accepted values. When asking an AI assistant to create or
modify an Exasim application, include links or excerpts from the relevant
Reference pages and ask it to preserve the documented interfaces.

See [Using AI Tools](../using-ai-tools/index.md) for prompt templates and
validation guidance.
