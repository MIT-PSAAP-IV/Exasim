# pyt2c

A single-stage, pip-`symengine` reimplementation of Exasim's `text2code` model
codegen: parse a `pdemodel.txt` and emit the header-only concrete model
`generated/my_model.hpp`.

The C++ `text2code` is a metaprogram — it *emits* a SymEngine C++ program,
compiles it, and runs it to produce `my_model.hpp`. `pyt2c` collapses those three
stages into one Python program. The only dependency is `pip install symengine`
(no vendored SymEngine build).

## Use

```sh
pip install symengine
python -m pyt2c path/to/pdemodel.txt -o generated       # writes generated/my_model.hpp
python -m pyt2c path/to/pdemodel.txt --stdout           # to stdout
```

```python
from pyt2c import parse_file, generate_header
print(generate_header(parse_file("pdemodel.txt")))
```

## Equivalence

`../tests/run_equiv.sh` compiles both the C++ `text2code` golden and the `pyt2c`
output into a numeric harness and diffs every kernel at a fixed input point. The
two are **numerically byte-identical** for the poisson2d and isoq2d compressible
Navier–Stokes (model 100) cases; the only textual difference is CSE temporary
ordering (the pip SymEngine and the vendored SymEngine order common-subexpression
elimination slightly differently, but the arithmetic is the same).

## Scope

- Emits `my_model.hpp` only (the concrete templated model). It does **not** build
  the legacy runtime-loaded `.so` model ABI — the header-only C++-driven app path
  does not need it.
- Mesh/`datain` preprocessing is still done by the C++ `text2code` (or the
  frontends); `pyt2c` is the model-codegen half.

## Layout

- `pyt2c/parser.py`  — `pdemodel.txt` DSL parser (mirrors `TextParser.hpp`).
- `pyt2c/interp.py`  — DSL body → `list[symengine.Expression]` interpreter.
- `pyt2c/codegen.py` — `my_model.hpp` emitter (mirrors `generateModelHeader`).
- `pyt2c/__main__.py`— CLI.
