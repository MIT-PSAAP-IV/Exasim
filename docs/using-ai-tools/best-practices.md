# Best Practices

Use AI tools as fast drafting and review assistants. Keep human control over
physics, numerical methods, validation, and final code acceptance.

## Recommended Workflow

1. Start from an existing Exasim example.
2. Provide the AI with exact file paths and relevant source files.
3. Specify geometry, units, parameters, boundary conditions, and outputs.
4. Ask for incremental changes.
5. Review assumptions before accepting edits.
6. Compile and run small cases frequently.
7. Validate against analytical, manufactured, or benchmark results.
8. Iterate with narrow follow-up prompts.

## Good Habits

- Tell the AI what not to change.
- Ask for affected files before coding.
- Ask for explicit assumptions.
- Ask for verification commands.
- Keep a clean git diff.
- Review generated array sizes and dimensions.
- Prefer simple examples before complex multiphysics.
- Preserve existing Exasim APIs unless intentionally extending them.

## Common Mistakes

| Mistake | Consequence |
| --- | --- |
| "Create a solver" without details | Underspecified physics and unusable code. |
| Missing boundary conditions | Ill-posed or incorrect PDE problem. |
| Missing units | Wrong scale, stiffness, or nondimensional parameters. |
| Too many changes at once | Hard-to-review diffs and hidden regressions. |
| No reference example | Wrong frontend style or obsolete API usage. |
| No validation criteria | Code may run but solve the wrong problem. |
| Ignoring GPU/MPI data movement | CPU-only code or stale device data bugs. |

## Best Prompt Style

Use direct, testable instructions:

```text
Modify only <file>. Preserve existing behavior unless <condition>. Add a small
test or example. Run <command>. Explain residual risks.
```

## When To Stop And Ask For Human Review

Request expert review when a change affects:

- conservation or consistency;
- HDG static condensation;
- nonlinear or linear solver logic;
- preconditioner storage;
- MPI communication;
- CUDA/HIP memory ownership;
- production physics models;
- benchmark results.
