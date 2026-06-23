# Using AI Tools

Modern AI coding assistants can accelerate Exasim application development when
they are given precise scientific, numerical, and repository context. Tools such
as OpenAI Codex, Claude, ChatGPT, GitHub Copilot, Gemini, and similar systems
can help draft examples, meshes, `pdeapp` files, `pdemodel` files, boundary
conditions, parameter sweeps, postprocessing scripts, tests, and documentation.

AI-generated code is not a substitute for numerical verification. Treat it as a
fast draft that must be reviewed, tested, and validated before production use.

## Why Use AI Tools?

AI tools are useful for Exasim tasks that are structured, file-based, and easy
to check:

| Task | How AI helps |
| --- | --- |
| Create examples | Copies a nearby example, modifies geometry/physics/options, and creates a new directory. |
| Implement PDE models | Drafts `ModelC`, `ModelD`, or `ModelW` callbacks from mathematical equations. |
| Generate meshes | Produces structured, boundary-layer, or geometry-specific mesh helper functions. |
| Write input files | Creates `pdeapp.m`, `pdeapp.py`, `pdeapp.jl`, `pdeapp.txt`, and `pdemodel.txt` drafts. |
| Add boundary/source/EOS terms | Converts mathematical descriptions into callback code. |
| Create parameter sweeps | Adds `physicsparamsweep`, output metadata, and warm-start settings. |
| Debug failures | Interprets compiler errors, runtime messages, residual histories, and mesh issues. |
| Extend Exasim | Drafts focused backend/frontend/doc patches for expert review. |
| Learn the framework | Explains how existing examples map to frontends, Text2Code, and backend data. |

## What AI Needs From You

Strong prompts provide the same information a human Exasim developer would need:

- the exact directory where work should happen;
- a nearby reference example to copy or adapt;
- the PDE model class (`ModelC`, `ModelD`, or `ModelW`) when known;
- geometry, dimensions, and units;
- boundary and initial conditions;
- material properties and `physicsparam` ordering;
- solver requirements, backend, MPI/GPU constraints, and expected outputs;
- validation criteria such as analytical solutions, benchmark values, or
  residual behavior.

## Recommended Workflow

```mermaid
flowchart LR
  Q["write precise prompt"] --> C["provide relevant files/examples"]
  C --> D["AI drafts focused changes"]
  D --> R["review code and assumptions"]
  R --> T["compile and run small case"]
  T --> V["validate physics and numerics"]
  V --> I["iterate with targeted prompts"]
```

Keep each iteration small. Ask for one example, one mesh function, one boundary
condition, or one test at a time. Large prompts that ask for a full solver,
GPU optimization, new preconditioner, and documentation in one step are harder
to review and more likely to produce incorrect code.

## Section Map

| Page | Use it for |
| --- | --- |
| [Prompt Engineering](prompt-engineering.md) | Writing precise, Exasim-specific prompts. |
| [Creating Examples](creating-examples.md) | Generating complete examples from existing ones. |
| [Implementing Physics Models](implementing-physics-models.md) | Drafting model callbacks, EOS, auxiliary variables, and PDE terms. |
| [Mesh Generation](mesh-generation.md) | Asking for structured, boundary-layer, curved, and geometry-specific meshes. |
| [Debugging](debugging.md) | Using AI for build errors, runtime failures, convergence issues, and review. |
| [Extending Exasim](extending-exasim.md) | Safely using AI for backend, frontend, Text2Code, GPU, and documentation changes. |
| [Validating AI-Generated Code](validating-ai-generated-code.md) | Checking correctness before relying on generated code. |
| [Best Practices](best-practices.md) | A concise workflow and common pitfalls. |
| [Example Prompts](example-prompts.md) | Copy-paste prompts and detailed case studies. |

## See Also

- [Quickstart](../getting-started/quickstart.md)
- [Frontends](../frontends/index.md)
- [Built-in library](../usage-modes/builtin.md)
- [Text2Code Applications](../text2code-applications/index.md)
- [Physics Models](../physics-models/index.md)
- [Theory](../theory/index.md)
- [Parameter sweeps](../usage-modes/parameter-sweeps.md)
- [Postprocessing](../usage-modes/postprocessing.md)
