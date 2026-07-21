# Benchmark: pyt2c vs C++ text2code

Task: `pdemodel.txt` → `generated/my_model.hpp` for the isoq2d compressible
Navier–Stokes model 100 (12 solution components, Sutherland viscosity, Riemann
boundary solver, full analytic Jacobians — the heaviest symbolic case in the
CHEFSI apps).

Machine: Apple Silicon (darwin), symengine pip wheel 0.14.1 (py3.14), g++ = clang.
Command: `bench/bench.py ... -n 5` (median of 5).

| Path | median |
| --- | --- |
| **pyt2c** (one-stage Python: parse → symengine → emit) | **62 ms** |
| C++ text2code (end-to-end: emit SymEngine program → g++ compile → run) | 1985 ms |
| &nbsp;&nbsp;— of which the g++ compile of `Code2Cpp.cpp` alone | 1601 ms |

**pyt2c is ~32× faster end-to-end, and ~26× faster than even just the g++ compile
step that the C++ approach cannot avoid.** The C++ pipeline's cost is dominated by
compiling a fresh SymEngine-heavy translation unit on every model; pyt2c does the
symbolic work directly in the interpreter, so there is no compile-a-program stage.

The pyt2c 62 ms includes Python interpreter startup + `import symengine` (~40 ms);
the actual parse+diff+CSE+print for this model is a small fraction of that. For
batch/repeated generation (e.g. a parameter study emitting many models in one
process) the per-model marginal cost is far below 62 ms.

Reproduce:

```sh
python bench/bench.py \
  --venv-python /path/to/venv/python \
  --pyt2c-dir  Exasim/text2code2/pyt2c \
  --text2code  /path/to/text2code \
  --pdeapp CASE/pdeapp.txt --pdemodel CASE/pdemodel.txt \
  --compile-cmd-file compilecmd.txt -n 5
```
