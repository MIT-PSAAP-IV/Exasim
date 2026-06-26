# Implementing Physics Models With AI

AI tools can help translate mathematical PDE descriptions into Exasim model
callbacks. They are most useful when the prompt names the model formulation and
defines the state layout explicitly.

Read [Physics Models](../physics-models/index.md) before asking AI to implement
a new model. The prompt should identify whether the application is `ModelC`,
`ModelD`, or `ModelW`.

## What AI Can Draft

- `flux` and `source` terms.
- `ubou`, `fbou`, `fbouhdg`, `fhat`, `uhat`, and `stab` callbacks.
- `initu`, `initq`, `initw`, and `initodg`.
- EOS helper functions.
- auxiliary equations through `w`.
- visualization callbacks such as `visscalars` and `visvectors`.
- QoI callbacks.
- Text2Code `pdemodel.txt` descriptions.

## ModelC Prompt Template

```text
Implement a ModelC Exasim model in <path>/pdemodel.m.

Geometry:
- domain:
- units:
- boundary tags and their physical meaning:

PDE:
- write the strong form here
- define u components
- define flux F(u,w,v,x,t;mu)
- define source S(u,w,v,x,t;mu)

Boundary conditions:
- ID 1:
- ID 2:

Initial condition:
- ...

Parameters:
- physicsparam = [ ... ] with this ordering:

Use <reference example> for callback names and coding style.
Do not change pdeapp.m except where dimensions must match the model.
```

## ModelD Prompt Template

```text
Implement a ModelD Exasim model in <path>/pdemodel.m.

Geometry:
- domain:
- units:
- boundary tags and their physical meaning:

Use q + grad(u) = 0 with uq = [u, q].
Define:
- ncu =
- nd =
- ncq = ncu * nd
- nc = ncu + ncq

PDE:
- flux F(u,q,w,v,x,t;mu)
- source S(u,q,w,v,x,t;mu)

Boundary conditions:
- ID 1:
- ID 2:

Initial conditions:
- u:
- q if needed:

Parameters:
- physicsparam ordering:

Reference:
- <path/to/ModelD/example>
```

## ModelW Prompt Template

```text
Implement a ModelW Exasim model in <path>/pdemodel.m for a wave problem.

Geometry:
- domain:
- units:
- boundary tags and their physical meaning:

State layout:
- u =
- q =

Equations:
- q_t + grad(u) = 0
- u_t + div(F(u,q)) = S(u,q)

Boundary conditions:
- ...

Initial conditions:
- ...

Use an existing WaveEquation example as the reference.
```

## EOS And Auxiliary Variables

For EOS, auxiliary variables, or external variables, ask the AI to state the
dependency graph before coding:

```text
Before editing, list which quantities depend on u, q, w, v, x, t, and
physicsparam. Then implement the EOS helper and update only the flux/source
callbacks that need it.
```

This reduces hidden coupling and makes review easier.

## Validation Requirements

For AI-generated physics code, validate:

- dimensional consistency of state arrays;
- sign conventions for gradients and fluxes;
- conservation properties;
- boundary-condition consistency;
- positivity of density, pressure, temperature, or other constrained states;
- convergence against an analytical, manufactured, or benchmark solution.
