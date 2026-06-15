"""Generate a text2code-compatible ``pdemodel.txt`` from the symbolic Python
frontend model.

The Python frontend normally drives code generation through the SymPy->C
flatten/CSE path (see ``Gencode.gencode`` and ``getccode``/``sympyassign``).
This module instead emits the higher-level text2code DSL (``pdemodel.txt``) so
that an *exported* app can regenerate its kernels from a portable text file via
the installed ``text2code`` tool, without re-running the Python frontend.

The emitter mirrors the per-function dispatch in ``Gencode.gencode`` (same
u/q split out of udg, same set of model functions) but uses a much simpler
SymPy printer: text2code function bodies accept plain infix C++/SymEngine
expressions, so ``mu[0]*uq[1]`` is exactly what we want. We do NOT use the
existing CSE/flatten path here.

Symbol renaming (frontend symbol -> text2code 0-based vector indexing):
    xdgK   -> x[K-1]      (coordinates)
    udgK   -> uq[K-1]     (u and q concatenated == uq)
    wdgK   -> w[K-1]
    odgK   -> v[K-1]
    uhgK   -> uhat[K-1]
    nlgK   -> n[K-1]
    tauK   -> tau[K-1]
    paramK -> mu[K-1]
    uinfK  -> eta[K-1]
    time   -> t
"""

import os
import re
import numpy
from importlib import import_module

import sympy
from sympy.printing.c import C99CodePrinter

from .syminit import syminit


# ---------------------------------------------------------------------------
# SymPy -> text2code printer
# ---------------------------------------------------------------------------

# frontend prefix -> text2code vector name
_SYMBOL_MAP = {
    "xdg": "x",
    "udg": "uq",
    "wdg": "w",
    "odg": "v",
    "uhg": "uhat",
    "nlg": "n",
    "tau": "tau",
    "param": "mu",
    "uinf": "eta",
}

# matches a frontend symbol like "udg12" -> ("udg", "12")
_SYM_RE = re.compile(r"^(" + "|".join(sorted(_SYMBOL_MAP, key=len, reverse=True)) + r")(\d+)$")


class Text2CodePrinter(C99CodePrinter):
    """Print SymPy expressions as text2code function-body C++/SymEngine infix.

    Differences from the stock C99 printer:
      * frontend symbols are renamed to 0-based vector indexing (see _SYMBOL_MAP),
        and the scalar ``time`` becomes ``t``;
      * ``pi`` prints as the bare token ``pi`` (text2code rewrites it to
        ``SymEngine::pi``), not ``M_PI`` which is a C double macro that cannot
        be used in an Expression context;
      * integer powers are expanded into repeated multiplication so we never
        emit ``pow(M_PI, 2)`` (M_PI again) or rely on ``pow`` over Expressions.
    """

    def _print_Symbol(self, expr):
        name = expr.name
        if name == "time":
            return "t"
        m = _SYM_RE.match(name)
        if m:
            base, idx = m.group(1), int(m.group(2))
            return "%s[%d]" % (_SYMBOL_MAP[base], idx - 1)
        return name

    def _print_Pi(self, expr):
        # Wrap in Expression(): bare SymEngine::pi is an RCP<const Constant>,
        # which has no operator* with int/Expression in the generated body.
        # Expression(pi) makes all surrounding arithmetic Expression-typed.
        return "Expression(pi)"

    def _print_Pow(self, expr):
        base, exp = expr.base, expr.exp
        # Expand small non-negative integer powers into repeated multiplication
        # to avoid pow()/M_PI; SymEngine handles a*a*a directly.
        if exp.is_Integer and exp > 0 and exp <= 8:
            b = self.parenthesize(base, sympy.printing.precedence.PRECEDENCE["Mul"])
            return "*".join([b] * int(exp))
        return super()._print_Pow(expr)


def _printer():
    return Text2CodePrinter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_uq(udg, ncu, nc):
    """Mirror gencode's u/q split out of the concatenated udg array."""
    u = udg[0:ncu]
    q = udg[ncu:] if nc > ncu else []
    return u, q


def _emit_function(name, args, output_name, exprs, printer):
    """Render one ``function NAME(args) ... end`` block.

    ``exprs`` is a flat 1-D iterable of SymPy expressions (one per output
    component); ``output_name`` is the text2code output vector name.
    """
    exprs = list(exprs)
    lines = []
    lines.append("function %s(%s)" % (name, ", ".join(args)))
    lines.append("  output_size(%s) = %d;" % (output_name, len(exprs)))
    for i, e in enumerate(exprs):
        lines.append("  %s[%d] = %s;" % (output_name, i, printer.doprint(sympy.sympify(e))))
    lines.append("end")
    return "\n".join(lines) + "\n"


def _flatten_F(f):
    """Flatten a numpy/sympy array in column-major order, like gencode."""
    arr = numpy.array(f)
    return arr.flatten("F")


# Argument lists used by text2code (must match the names declared in the
# scalars/vectors header). Element (volume) functions take the field args;
# face/boundary functions additionally take uhat, n, tau.
_ELEM_ARGS = ["x", "uq", "v", "w", "eta", "mu", "t"]
_FACE_ARGS = ["x", "uq", "v", "w", "uhat", "n", "tau", "eta", "mu", "t"]
_INIT_ARGS = ["x", "eta", "mu"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _header(app):
    nd = app["nd"]
    nc = app["nc"]
    ncx = app["ncx"]
    ncu = app["ncu"]
    nco = app["nco"]
    ncw = app["ncw"]
    ntau = len(app["tau"])
    nuinf = len(app["uinf"])
    nparam = len(app["physicsparam"])
    next_ = len(app.get("externalparam", [0.0])) or 1

    lines = []
    lines.append("scalars t")
    lines.append("")
    lines.append(
        "vectors x(%d), uq(%d), v(%d), w(%d), uhat(%d), uext(%d), n(%d), tau(%d), mu(%d), eta(%d)"
        % (ncx, nc, nco, ncw, ncu, next_, nd, ntau, nparam, nuinf)
    )
    lines.append("")
    lines.append("jacobian uq, w, uhat")
    lines.append("")
    lines.append("hessian")
    lines.append("")
    lines.append("batch x, uq, v, w, uhat, n, uext")
    lines.append("")
    return lines


def _outputs_line(present):
    """The ``outputs`` declaration, ordered to match text2code's canonical
    first-six requirement (Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg) followed
    by any optional functions actually emitted."""
    order = [
        "Flux", "Source", "Tdfunc", "Ubou", "Fbou", "FbouHdg",
        "Initu", "VisScalars", "VisVectors", "VisTensors",
        "QoIvolume", "QoIboundary",
    ]
    outs = [o for o in order if o in present]
    return "outputs " + ", ".join(outs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def genpdemodel(pde, dest_path):
    """Write a text2code-compatible ``pdemodel.txt`` for ``pde`` to ``dest_path``.

    ``pde`` is the frontend app dict (already preprocessed, so dimensions such
    as nc/ncu/ncq/nco/ncw are set). The symbolic model is ``pde['modelfile']``.

    Covers the kernel set text2code requires plus the common optional ones:
    Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg, Initu, VisScalars, VisVectors,
    VisTensors, QoIvolume, QoIboundary.

    Raises if a required model function (flux, fbou, ubou, fbouhdg, initu) is
    missing, mirroring gencode's hard requirements.
    """
    app = pde
    xdg, udg, _udg1, _udg2, wdg, _wdg1, _wdg2, odg, _odg1, _odg2, uhg, nlg, tau, uinf, param, time = syminit(app)[0:16]

    model = import_module(app["modelfile"])

    nc = app["nc"]
    ncu = app["ncu"]
    nd = app["nd"]
    u, q = _split_uq(udg, ncu, nc)

    pr = _printer()
    blocks = []
    present = set()

    def call(fn, *a):
        return getattr(model, fn)(*a)

    # ----- Flux (required) -----
    if not hasattr(model, "flux"):
        raise RuntimeError("genpdemodel: pde.flux is not defined")
    f = call("flux", u, q, wdg, odg, xdg, time, param, uinf)
    blocks.append(_emit_function("Flux", _ELEM_ARGS, "f", _flatten_F(f), pr))
    present.add("Flux")

    # ----- Source (required by text2code; default to zero if absent) -----
    if hasattr(model, "source"):
        f = call("source", u, q, wdg, odg, xdg, time, param, uinf)
        blocks.append(_emit_function("Source", _ELEM_ARGS, "s", _flatten_F(f), pr))
    else:
        blocks.append(_emit_function("Source", _ELEM_ARGS, "s",
                                     [sympy.Integer(0)] * ncu, pr))
    present.add("Source")

    # ----- Tdfunc (mass; required by text2code; default to ones if absent) -----
    if hasattr(model, "mass"):
        f = call("mass", u, q, wdg, odg, xdg, time, param, uinf)
        blocks.append(_emit_function("Tdfunc", _ELEM_ARGS, "m", _flatten_F(f), pr))
    else:
        blocks.append(_emit_function("Tdfunc", _ELEM_ARGS, "m",
                                     [sympy.Integer(1)] * ncu, pr))
    present.add("Tdfunc")

    # ----- Ubou (required) -----
    if not hasattr(model, "ubou"):
        raise RuntimeError("genpdemodel: pde.ubou is not defined")
    f = call("ubou", u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
    blocks.append(_emit_function("Ubou", _FACE_ARGS, "ub", _flatten_F(f), pr))
    present.add("Ubou")

    # ----- Fbou (required) -----
    if not hasattr(model, "fbou"):
        raise RuntimeError("genpdemodel: pde.fbou is not defined")
    f = call("fbou", u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
    blocks.append(_emit_function("Fbou", _FACE_ARGS, "fb", _flatten_F(f), pr))
    present.add("Fbou")

    # ----- FbouHdg (required for hybrid) -----
    if not hasattr(model, "fbouhdg"):
        raise RuntimeError("genpdemodel: pde.fbouhdg is not defined")
    f = call("fbouhdg", u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
    blocks.append(_emit_function("FbouHdg", _FACE_ARGS, "fb", _flatten_F(f), pr))
    present.add("FbouHdg")

    # ----- Initu (required) -----
    if not hasattr(model, "initu"):
        raise RuntimeError("genpdemodel: pde.initu is not defined")
    f = model.initu(xdg, param, uinf)
    blocks.append(_emit_function("Initu", _INIT_ARGS, "ui", _flatten_F(f), pr))
    present.add("Initu")

    # ----- optional visualization / QoI functions -----
    if hasattr(model, "visscalars"):
        f = call("visscalars", u, q, wdg, odg, xdg, time, param, uinf)
        blocks.append(_emit_function("VisScalars", _ELEM_ARGS, "s", _flatten_F(f), pr))
        present.add("VisScalars")
    if hasattr(model, "visvectors"):
        f = call("visvectors", u, q, wdg, odg, xdg, time, param, uinf)
        blocks.append(_emit_function("VisVectors", _ELEM_ARGS, "s", _flatten_F(f), pr))
        present.add("VisVectors")
    if hasattr(model, "vistensors"):
        f = call("vistensors", u, q, wdg, odg, xdg, time, param, uinf)
        blocks.append(_emit_function("VisTensors", _ELEM_ARGS, "s", _flatten_F(f), pr))
        present.add("VisTensors")
    if hasattr(model, "qoivolume"):
        f = call("qoivolume", u, q, wdg, odg, xdg, time, param, uinf)
        blocks.append(_emit_function("QoIvolume", _ELEM_ARGS, "s", _flatten_F(f), pr))
        present.add("QoIvolume")
    if hasattr(model, "qoiboundary"):
        f = call("qoiboundary", u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
        blocks.append(_emit_function("QoIboundary", _FACE_ARGS, "fb", _flatten_F(f), pr))
        present.add("QoIboundary")

    # ----- assemble -----
    out = []
    out.extend(_header(app))
    out.append(_outputs_line(present))
    out.append("")
    out.append("\n".join(blocks))

    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"

    with open(dest_path, "w") as fid:
        fid.write(text)

    return dest_path
