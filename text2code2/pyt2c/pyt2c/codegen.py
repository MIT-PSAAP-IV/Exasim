"""Emit ``generated/my_model.hpp`` from a parsed model + evaluated functions.

Python equivalent of ``SymbolicScalarsVectors::generateModelHeader``. Reproduces
the production emitter (matching the isoq2d model100 golden): the size block +
optional external-coupling constants, the volume/boundary/init value methods,
their column-major Jacobians, and the Fint/Fext value+Jacobian methods.
"""
from __future__ import annotations

import re

import symengine as se

from .parser import Spec, FunctionDef
from .interp import ModelEvaluator


try:
    from symengine import ccode as _ccode
except ImportError:  # pragma: no cover
    from symengine.lib.symengine_wrapper import ccode as _ccode


_MATH_RE = re.compile(
    r"(\b(?:pow|sqrt|exp|log|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|fabs|atan2)\b)(?=\s*\()"
)


def kokkosify(s: str) -> str:
    """Prefix C99 math intrinsics with ``Kokkos::`` (matches the C++ regex)."""
    return _MATH_RE.sub(r"Kokkos::\1", s)


def cxx(expr) -> str:
    return kokkosify(_ccode(expr))


def _rename_input(name: str) -> str:
    if name == "eta":
        return "uinf"
    if name == "uhat":
        return "uh"
    return name


# Signatures (verbatim from generateModelHeader).
VOLUME_SIG = ("dstype f[], const dstype x[], const dstype uq[], const dstype v[], "
              "const dstype w[], const dstype mu[], const dstype uinf[], dstype t")
INITU_SIG = "dstype f[], const dstype x[], const dstype uinf[], const dstype mu[]"
BOUNDARY_SIG = ("dstype f[], int ib, const dstype x[], const dstype uq[], const dstype v[],"
                " const dstype w[], const dstype uh[], const dstype n[], const dstype tau[],"
                " const dstype mu[], const dstype uinf[], dstype t")
FEXT_SIG = ("dstype f[], int ib, const dstype x[], const dstype uq[], const dstype v[],"
            " const dstype w[], const dstype uh[], const dstype n[], const dstype uext[],"
            " const dstype tau[], const dstype mu[], const dstype uinf[], dstype t")

VOLUME_METHODS = [
    ("Flux", "flux"), ("Source", "source"), ("Tdfunc", "tdfunc"),
    ("VisScalars", "vis_scalars"), ("VisVectors", "vis_vectors"),
    ("QoIvolume", "qoi_volume"),
]
BOUNDARY_METHODS = [
    ("Fbou", "fbou"), ("Ubou", "ubou"), ("FbouHdg", "fbou_hdg"),
    ("QoIboundary", "qoi_boundary"),
]
VOLUME_JAC = [
    ("Flux", ["flux_jac_uq", "flux_jac_w"]),
    ("Source", ["source_jac_uq", "source_jac_w"]),
]
BOUNDARY_JAC = [
    ("FbouHdg", ["fbou_hdg_jac_uq", "fbou_hdg_jac_w", "fbou_hdg_jac_uh"]),
    ("Fbou", ["fbou_jac_uq", "fbou_jac_w", "fbou_jac_uh"]),
    ("Ubou", ["ubou_jac_uq", "ubou_jac_w", "ubou_jac_uh"]),
]


class HeaderGenerator:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.ev = ModelEvaluator(spec)
        self.szuhat = spec.vectors.get("uhat", 0)

    # -- per-function metadata ------------------------------------------------
    def input_vectors(self, fdef: FunctionDef):
        """Ordered (argname, symbol-vector) for the vector args of a function."""
        out = []
        for arg in fdef.args:
            if arg in self.spec.vectors:
                out.append((arg, self.ev.vectors[arg]))
        return out

    def jac_inputs(self, fdef: FunctionDef):
        """Vector args (in arg order) that also appear in the `jacobian` list."""
        return [
            self.ev.vectors[arg]
            for arg in fdef.args
            if arg in self.spec.vectors and arg in self.spec.jacobian
        ]

    @staticmethod
    def diff_to_exprs(f, inp):
        """Column-major flatten: J[j*nf + i] = d f[i] / d inp[j]."""
        result = []
        for j in range(len(inp)):
            for i in range(len(f)):
                result.append(se.diff(f[i], inp[j]))
        return result

    # -- emitters -------------------------------------------------------------
    def emit_value(self, out, method: str, sig: str, f, fdef: FunctionDef, indent="    "):
        out.append(f"{indent}KOKKOS_INLINE_FUNCTION static")
        out.append(f"{indent}void {method}({sig}) {{")
        if len(f) == 0:
            out.append(f"{indent}    // empty body — defaulted via ModelDefaults")
            out.append(f"{indent}}}\n")
            return
        self._emit_body(out, f, fdef, indent + "    ")
        out.append(f"{indent}}}\n")

    def emit_value_per_ib(self, out, method: str, sig: str, f, fdef: FunctionDef, szblk: int):
        out.append("    KOKKOS_INLINE_FUNCTION static")
        out.append(f"    void {method}({sig}) {{")
        nbc = (len(f) // szblk) if szblk > 0 else 0
        for nb in range(nbc):
            g = [f[m + nb * szblk] for m in range(szblk)]
            kw = "if" if nb == 0 else "else if"
            out.append(f"        {kw} (ib == {nb + 1}) {{")
            self._emit_body(out, g, fdef, "            ")
            out.append("        }")
        out.append("    }\n")

    def _emit_body(self, out, f, fdef: FunctionDef, ind: str):
        used = set()
        for e in f:
            used |= set(se.sympify(e).free_symbols)
        used_names = {str(s) for s in used}

        repl, reduced = se.cse(f)

        for name, vec in self.input_vectors(fdef):
            arr = _rename_input(name)
            for j, sym in enumerate(vec):
                if str(sym) in used_names:
                    out.append(f"{ind}const dstype {name}{j} = {arr}[{j}];")
        if repl:
            out.append("")
        for sym, expr in repl:
            out.append(f"{ind}const dstype {sym} = {cxx(expr)};")
        out.append("")
        for n, r in enumerate(reduced):
            out.append(f"{ind}f[{n}] = {cxx(r)};")

    # -- top level ------------------------------------------------------------
    def generate(self) -> str:
        s = self.spec
        v = s.vectors
        nd = v.get("x", 0)
        ncu = v.get("uhat", 0)
        ncw = v.get("w", 0)
        nco = v.get("v", 0)
        nparam = v.get("mu", 0)
        ntau = v.get("tau", 0)

        out: list[str] = []
        out.append("// Auto-generated by text2code (pyt2c). Do not edit by hand.")
        out.append("// Regenerate with `python -m pyt2c <pdemodel.txt> -o <dir>`.")
        out.append("//")
        out.append("// This header is consumed by `<exasim/model.hpp>`'s templated FEM")
        out.append("// internals. The struct below satisfies the Model contract; the")
        out.append("// inherited `ModelDefaults<GeneratedModel>` supplies zero-fill")
        out.append("// defaults for any optional method this PDE doesn't define.")
        out.append("#pragma once\n")
        out.append("#include <Kokkos_Core.hpp>")
        out.append("struct PdeModel : ModelDefaults<PdeModel> {")
        out.append(f"    static constexpr int nd     = {nd};")
        out.append(f"    static constexpr int ncu    = {ncu};")
        out.append(f"    static constexpr int ncw    = {ncw};")
        out.append(f"    static constexpr int nco    = {nco};")
        out.append(f"    static constexpr int nparam = {nparam};")
        out.append(f"    static constexpr int ntau   = {ntau};")

        # Visualization / QoI output-size metadata (provider paths read these off PdeModel).
        # Matches the C++ text2code generateModelSizesHpp: from the declared output_size of
        # the Vis*/QoI functions. nvec/nten are per-point counts (divide out the nd / nd*nd
        # spatial components).
        def _osz(name):
            f = s.function(name)
            return f.outputsize if (f is not None and s.is_output(name)) else 0
        nsca = _osz("VisScalars")
        nvec = (_osz("VisVectors") // nd) if nd else 0
        nten = (_osz("VisTensors") // (nd * nd)) if nd else 0
        nsurf = _osz("QoIboundary")
        nvqoi = _osz("QoIvolume")
        out.append(f"    static constexpr int nsca   = {nsca};")
        out.append(f"    static constexpr int nvec   = {nvec};")
        out.append(f"    static constexpr int nten   = {nten};")
        out.append(f"    static constexpr int nsurf  = {nsurf};")
        out.append(f"    static constexpr int nvqoi  = {nvqoi};")
        out.append("    static constexpr int Nq = ncu * (1 + nd);")

        has_fint = s.is_output("Fint")
        has_fext = s.is_output("Fext")
        if has_fint or has_fext:
            out.append("")
            out.append("    static constexpr bool has_external_coupling = true;")
            if has_fint:
                out.append(f"    static constexpr int nfint  = {len(self.ev.evaluate('Fint'))};")
            if has_fext:
                out.append(f"    static constexpr int nfext  = {len(self.ev.evaluate('Fext'))};")
            nuext = v.get("uext", 0)
            if nuext > 0:
                out.append(f"    static constexpr int ncuext = {nuext};")
        out.append("")

        # ---- value methods ----
        for fname, method in VOLUME_METHODS:
            fdef = self._present(fname)
            if fdef:
                self.emit_value(out, method, VOLUME_SIG, self.ev.evaluate(fname), fdef)
        fdef = self._present("Initu")
        if fdef:
            self.emit_value(out, "initu", INITU_SIG, self.ev.evaluate("Initu"), fdef)
        for fname, method in BOUNDARY_METHODS:
            fdef = self._present(fname)
            if fdef:
                self.emit_value_per_ib(out, method, BOUNDARY_SIG,
                                       self.ev.evaluate(fname), fdef, self.szuhat)

        # ---- volume Jacobians ----
        for fname, jac_names in VOLUME_JAC:
            fdef = self._present(fname)
            if not fdef:
                continue
            f = self.ev.evaluate(fname)
            for k, inp in enumerate(self.jac_inputs(fdef)):
                if not inp or k >= len(jac_names):
                    continue
                jac = self.diff_to_exprs(f, inp)
                self.emit_value(out, jac_names[k], VOLUME_SIG, jac, fdef)

        # ---- boundary Jacobians (per-ib, widened block) ----
        for fname, jac_names in BOUNDARY_JAC:
            fdef = self._present(fname)
            if not fdef:
                continue
            f = self.ev.evaluate(fname)
            nbc = (len(f) // self.szuhat) if self.szuhat > 0 else 0
            for k, inp in enumerate(self.jac_inputs(fdef)):
                if not inp or k >= len(jac_names):
                    continue
                jblock = self.szuhat * len(inp)
                jac_all = []
                for nb in range(nbc):
                    g = [f[m + nb * self.szuhat] for m in range(self.szuhat)]
                    jac_all.extend(self.diff_to_exprs(g, inp))
                self.emit_value_per_ib(out, jac_names[k], BOUNDARY_SIG, jac_all, fdef, jblock)

        # ---- Fint / Fext value + Jacobians (plain single block) ----
        for fname, method, sig, jnames in [
            ("Fint", "fint", BOUNDARY_SIG, ["fint_jac_uq", "fint_jac_w", "fint_jac_uh"]),
            ("Fext", "fext", FEXT_SIG, ["fext_jac_uq", "fext_jac_w", "fext_jac_uh"]),
        ]:
            fdef = self._present(fname)
            if not fdef:
                continue
            f = self.ev.evaluate(fname)
            self.emit_value(out, method, sig, f, fdef)
            for k, inp in enumerate(self.jac_inputs(fdef)):
                if not inp or k >= len(jnames):
                    continue
                jac = self.diff_to_exprs(f, inp)
                self.emit_value(out, jnames[k], sig, jac, fdef)

        out.append("};")
        return "\n".join(out) + "\n"

    def _present(self, fname: str) -> FunctionDef | None:
        fdef = self.spec.function(fname)
        if fdef is None or not self.spec.is_output(fname):
            return None
        return fdef


def generate_header(spec: Spec) -> str:
    return HeaderGenerator(spec).generate()
