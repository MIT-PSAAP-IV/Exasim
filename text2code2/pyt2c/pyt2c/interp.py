"""Interpreter that evaluates a ``pdemodel.txt`` function body into a list of
SymEngine expressions.

This is the Python equivalent of the ``SymbolicFunctions.cpp`` the C++ text2code
emits: each model function, when evaluated against the global symbol vectors,
returns ``list[Expression]`` of length ``output_size``. Assignments are executed
with Python ``exec`` against a namespace of math/matrix helpers; ``for``/``endfor``
loops, ``matrix``/``vector`` declarations and ``zeros``/``ones``/``fill`` are
handled explicitly.
"""
from __future__ import annotations

import re

import symengine as se

from .parser import Spec, FunctionDef


# ---------------------------------------------------------------------------
# Matrix wrapper: supports K[i][j] get/set (row proxy), matmul, scalar mul, add.
# ---------------------------------------------------------------------------
class _Row:
    def __init__(self, mat: "Mat", i: int):
        self._mat = mat
        self._i = int(i)

    def __getitem__(self, j):
        return self._mat.m[self._i, int(j)]

    def __setitem__(self, j, val):
        self._mat.m[self._i, int(j)] = se.sympify(val)


class Mat:
    def __init__(self, rows=None, cols=None, dense=None):
        if dense is not None:
            self.m = dense
        else:
            self.m = se.DenseMatrix(int(rows), int(cols),
                                    [se.Integer(0)] * (int(rows) * int(cols)))

    def __getitem__(self, i):
        return _Row(self, i)

    def __mul__(self, other):
        if isinstance(other, Mat):
            return Mat(dense=self.m * other.m)
        return Mat(dense=self.m * se.sympify(other))

    def __rmul__(self, other):
        # scalar * matrix (scalar mul commutes)
        return Mat(dense=self.m * se.sympify(other))

    def __add__(self, other):
        return Mat(dense=self.m + other.m)

    def __sub__(self, other):
        return Mat(dense=self.m - other.m)


# ---------------------------------------------------------------------------
# Math + matrix helper namespace exposed to the DSL body.
# ---------------------------------------------------------------------------
def _make_namespace():
    def _log10(x):
        return se.log(se.sympify(x)) / se.log(se.Integer(10))

    ns = {
        "sin": se.sin, "cos": se.cos, "tan": se.tan,
        "asin": se.asin, "acos": se.acos, "atan": se.atan, "atan2": se.atan2,
        "sinh": se.sinh, "cosh": se.cosh, "tanh": se.tanh,
        "asinh": se.asinh, "acosh": se.acosh, "atanh": se.atanh,
        "exp": se.exp, "log": se.log, "log10": _log10,
        "sqrt": se.sqrt, "pow": lambda a, b: se.sympify(a) ** se.sympify(b),
        "abs": se.Abs, "fabs": se.Abs,
        "erf": se.erf, "erfc": se.erfc,
        "pi": se.pi,
        "Expression": lambda v: se.sympify(v),
        "mul": lambda a, b: (a * b),
        "inv": lambda M: Mat(dense=M.m.inv()),
        "transpose": lambda M: Mat(dense=M.m.transpose()),
        "det": lambda M: M.m.det(),
        "trace": lambda M: M.m.trace(),
    }
    return ns


_FOR_RE = re.compile(r"^\s*for\s+(\w+)\s+in\s+([^:]+):(.+?)\s*$")
_MATRIX_DECL_RE = re.compile(r"^\s*matrix\s+(\w+)\((\d+),(\d+)\)\s*;?\s*$")
_VECTOR_DECL_RE = re.compile(r"^\s*vector\s+(\w+)\((\d+)\)\s*;?\s*$")
_OUTPUT_SIZE_RE = re.compile(r"^\s*output_size\((\w+)\)\s*=\s*(\d+)\s*;?\s*$")
_FILL_RE = re.compile(r"^\s*(zeros|ones|fill)\s*\((.*)\)\s*;?\s*$")


def _strip(line: str) -> str:
    line = line.split("//", 1)[0]
    line = line.rstrip()
    if line.endswith(";"):
        line = line[:-1]
    return line.rstrip()


class ModelEvaluator:
    """Evaluates model functions against a shared set of global symbols."""

    def __init__(self, spec: Spec):
        self.spec = spec
        self.scalars = {name: se.Symbol(name) for name in spec.scalars}
        self.vectors = {
            name: [se.Symbol(f"{name}{j}") for j in range(size)]
            for name, size in spec.vectors.items()
        }
        self._base_ns = _make_namespace()
        # model functions callable from within a body (return their output list)
        for f in spec.functions:
            self._base_ns[f.name] = self._make_callable(f.name)
        self._cache: dict[str, list] = {}

    def _make_callable(self, name: str):
        # A body may call another model function with NON-standard arguments, e.g.
        # `ui = Vortex(x, t0c, mu)` binds Vortex's time param to the constant t0c
        # (not the global `t` symbol). So bind the called function's params to the
        # PASSED arguments positionally and evaluate its body under that binding.
        def _call(*args):
            fdef = self.spec.function(name)
            if fdef is None:
                raise KeyError(f"function {name!r} not found in model")
            env: dict = {}
            for param, val in zip(fdef.args, args):
                env[param] = list(val) if isinstance(val, list) else val
            return self._eval_body(fdef, env)
        return _call

    def evaluate(self, name: str) -> list:
        """Top-level evaluation: bind the function's args to the global symbols."""
        if name in self._cache:
            return self._cache[name]
        fdef = self.spec.function(name)
        if fdef is None:
            raise KeyError(f"function {name!r} not found in model")
        env: dict = {}
        for arg in fdef.args:
            if arg in self.scalars:
                env[arg] = self.scalars[arg]
            elif arg in self.vectors:
                env[arg] = list(self.vectors[arg])
            else:
                raise ValueError(
                    f"function {name}: argument {arg!r} is neither a declared "
                    "scalar nor vector"
                )
        out = self._eval_body(fdef, env)
        self._cache[name] = out
        return out

    def _eval_body(self, fdef: FunctionDef, env: dict) -> list:
        stmts = [_strip(b) for b in fdef.body]
        stmts = [s for s in stmts if s != "" and s != "end"]
        self._run(stmts, env, fdef)
        if fdef.output not in env:
            raise ValueError(f"function {fdef.name}: output {fdef.output!r} never set")
        return list(env[fdef.output])

    # ------------------------------------------------------------------
    def _run(self, stmts: list[str], env: dict, fdef: FunctionDef):
        i = 0
        n = len(stmts)
        while i < n:
            s = stmts[i]
            fm = _FOR_RE.match(s)
            if fm:
                var = fm.group(1)
                a = int(self._eval(fm.group(2), env))
                b = int(self._eval(fm.group(3), env))
                j = self._find_endfor(stmts, i)
                body = stmts[i + 1:j]
                for k in range(a, b + 1):
                    env[var] = k   # a plain int, so `arr[i]` indexing works
                    self._run(body, env, fdef)
                i = j + 1
                continue
            i += 1
            m = _OUTPUT_SIZE_RE.match(s)
            if m:
                env[m.group(1)] = [se.Integer(0)] * int(m.group(2))
                continue
            m = _MATRIX_DECL_RE.match(s)
            if m:
                env[m.group(1)] = Mat(int(m.group(2)), int(m.group(3)))
                continue
            m = _VECTOR_DECL_RE.match(s)
            if m:
                env[m.group(1)] = [se.Integer(0)] * int(m.group(2))
                continue
            m = _FILL_RE.match(s)
            if m:
                self._fill(m.group(1), m.group(2), env)
                continue
            # otherwise an assignment statement — exec against the namespace
            self._exec(s, env)

    @staticmethod
    def _find_endfor(stmts: list[str], start: int) -> int:
        depth = 0
        for j in range(start + 1, len(stmts)):
            if _FOR_RE.match(stmts[j]):
                depth += 1
            elif stmts[j].strip() == "endfor":
                if depth == 0:
                    return j
                depth -= 1
        raise ValueError("for without matching endfor")

    def _fill(self, kind: str, argstr: str, env: dict):
        args = [a.strip() for a in argstr.split(",")]
        name = args[0]
        if kind == "zeros":
            val = se.Integer(0)
        elif kind == "ones":
            val = se.Integer(1)
        else:  # fill
            val = self._eval(args[-1], env)
        if len(args) >= 2 and kind in ("zeros", "ones", "fill"):
            # explicit size given (zeros(v,N)/ones(v,N)/fill(v,N,val) or fill(v,val))
            try:
                size = int(self._eval(args[1], env))
                env[name] = [val] * size
                return
            except (ValueError, TypeError):
                pass
        # single-arg form: fill the existing (output) list in place
        cur = env.get(name)
        length = len(cur) if cur is not None else 0
        env[name] = [val] * length

    def _eval(self, expr: str, env: dict):
        return eval(expr, self._base_ns, env)  # noqa: S307 - controlled namespace

    def _exec(self, stmt: str, env: dict):
        exec(stmt, self._base_ns, env)  # noqa: S102 - controlled namespace
