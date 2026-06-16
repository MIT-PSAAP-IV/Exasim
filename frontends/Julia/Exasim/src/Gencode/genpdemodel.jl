# Generate a text2code-compatible `pdemodel.txt` from the symbolic Julia
# frontend model.
#
# This is a Julia port of the Python frontend's Gencode/genpdemodel.py. The
# Julia frontend normally drives code generation through the SymPy->C
# flatten/CSE path (see Gencode.gencode and getccode/sympyassign). This module
# instead emits the higher-level text2code DSL (pdemodel.txt) so that an
# *exported* app can regenerate its kernels from a portable text file via the
# installed text2code tool, without re-running the Julia frontend.
#
# The emitter mirrors the per-function dispatch in Gencode.gencode (same u/q
# split out of udg, same set of model functions) but uses a much simpler SymPy
# printer: text2code function bodies accept plain infix C++/SymEngine
# expressions, so `mu[0]*uq[1]` is exactly what we want. We do NOT use the
# existing CSE/flatten path here.
#
# Both frontends use the same SymPy underneath; the Text2CodePrinter is the
# identical sympy C99CodePrinter subclass, reached here through SymPy.PyCall, so
# symbol naming and renaming match the Python output byte-for-byte.
#
# Symbol renaming (frontend symbol -> text2code 0-based vector indexing):
#     xdgK   -> x[K-1]      (coordinates)
#     udgK   -> uq[K-1]     (u and q concatenated == uq)
#     wdgK   -> w[K-1]
#     odgK   -> v[K-1]
#     uhgK   -> uhat[K-1]
#     nlgK   -> n[K-1]
#     tauK   -> tau[K-1]
#     paramK -> mu[K-1]
#     uinfK  -> eta[K-1]
#     time   -> t

# ---------------------------------------------------------------------------
# SymPy -> text2code printer (defined once in the Python interpreter via PyCall)
# ---------------------------------------------------------------------------

# Python source for the Text2CodePrinter, kept identical to the validated
# Python frontend printer (same _print_Symbol / _print_Pi / _print_Pow). It is
# exec'd into a private namespace through PyCall at first use; we avoid the
# `py"..."` string macro because that resolves its module path at parse time,
# whereas we want a plain runtime exec.
const _T2C_PYSRC = raw"""
import re as _t2c_re
import sympy as _t2c_sympy
from sympy.printing.c import C99CodePrinter as _t2c_C99CodePrinter

# frontend prefix -> text2code vector name
_t2c_SYMBOL_MAP = {
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
_t2c_SYM_RE = _t2c_re.compile(
    r"^(" + "|".join(sorted(_t2c_SYMBOL_MAP, key=len, reverse=True)) + r")(\d+)$")


class Text2CodePrinter(_t2c_C99CodePrinter):
    # Print SymPy expressions as text2code function-body C++/SymEngine infix.
    #
    # Differences from the stock C99 printer:
    #   * frontend symbols are renamed to 0-based vector indexing
    #     (see _t2c_SYMBOL_MAP), and the scalar `time` becomes `t`;
    #   * `pi` prints as `Expression(pi)` (text2code uses SymEngine, where a
    #     bare `pi` is an RCP<const Constant> with no arithmetic operators);
    #   * small non-negative integer powers are expanded into repeated
    #     multiplication so we never emit `pow(M_PI, 2)` or rely on `pow`
    #     over Expressions.

    def _print_Symbol(self, expr):
        name = expr.name
        if name == "time":
            return "t"
        m = _t2c_SYM_RE.match(name)
        if m:
            base, idx = m.group(1), int(m.group(2))
            return "%s[%d]" % (_t2c_SYMBOL_MAP[base], idx - 1)
        return name

    def _print_Pi(self, expr):
        return "Expression(pi)"

    def _print_Pow(self, expr):
        base, exp = expr.base, expr.exp
        if exp.is_Integer and exp > 0 and exp <= 8:
            b = self.parenthesize(
                base, _t2c_sympy.printing.precedence.PRECEDENCE["Mul"])
            return "*".join([b] * int(exp))
        return super()._print_Pow(expr)


def _t2c_print(e):
    return Text2CodePrinter().doprint(_t2c_sympy.sympify(e))
"""

# Lazily install the Python-side Text2CodePrinter and return the print helper.
let _printer = Ref{Any}(nothing)
    global function _t2c_printer()
        if _printer[] === nothing
            PC = SymPy.PyCall
            ns = PC.pycall(PC.pybuiltin("dict"), PC.PyObject)
            PC.pycall(PC.pybuiltin("exec"), PC.PyObject, _T2C_PYSRC, ns)
            _printer[] = get(ns, "_t2c_print")
        end
        return _printer[]
    end
end

# Print a single SymPy.jl Sym (or plain number) as text2code infix.
function _t2c(printer, e)
    if isa(e, Sym)
        return printer(e.o)
    end
    # plain Julia number (e.g. Int/Float introduced by a zero/one default)
    return printer(Sym(e).o)
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mirror gencode's u/q split out of the concatenated udg array.
function _split_uq(udg, ncu, nc)
    u = udg[1:ncu]
    q = nc > ncu ? udg[ncu+1:end] : []
    return u, q
end

# Flatten a model function result into a flat Vector, mirroring gencode's
# `length(f)==1 -> reshape; f[:]` (column-major) pattern.
function _flatten_F(f)
    if length(f) == 1
        return [f]
    end
    return collect(f[:])
end

# Render one `function NAME(args) ... end` block. `exprs` is a flat iterable of
# expressions (one per output component); `output_name` is the text2code output
# vector name.
function _emit_function(name, args, output_name, exprs, printer)
    exprs = collect(exprs)
    lines = String[]
    push!(lines, "function $name($(join(args, ", ")))")
    push!(lines, "  output_size($output_name) = $(length(exprs));")
    for (i, e) in enumerate(exprs)
        push!(lines, "  $output_name[$(i-1)] = $(_t2c(printer, e));")
    end
    push!(lines, "end")
    return join(lines, "\n") * "\n"
end

# Argument lists used by text2code (must match the names declared in the
# scalars/vectors header). Element (volume) functions take the field args;
# face/boundary functions additionally take uhat, n, tau.
const _ELEM_ARGS = ["x", "uq", "v", "w", "eta", "mu", "t"]
const _FACE_ARGS = ["x", "uq", "v", "w", "uhat", "n", "tau", "eta", "mu", "t"]
const _INIT_ARGS = ["x", "eta", "mu"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

function _header(app)
    nd = app.nd
    nc = app.nc
    ncx = app.ncx
    ncu = app.ncu
    nco = app.nco
    ncw = app.ncw
    ntau = length(app.tau)
    nuinf = length(app.uinf)
    nparam = length(app.physicsparam)
    next = max(length(app.externalparam), 1)

    lines = String[]
    push!(lines, "scalars t")
    push!(lines, "")
    push!(lines, "vectors x($ncx), uq($nc), v($nco), w($ncw), uhat($ncu), " *
                 "uext($next), n($nd), tau($ntau), mu($nparam), eta($nuinf)")
    push!(lines, "")
    push!(lines, "jacobian uq, w, uhat")
    push!(lines, "")
    push!(lines, "hessian")
    push!(lines, "")
    push!(lines, "batch x, uq, v, w, uhat, n, uext")
    push!(lines, "")
    return lines
end

# The `outputs` declaration, ordered to match text2code's canonical first-six
# requirement (Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg) followed by any
# optional functions actually emitted.
function _outputs_line(present)
    order = ["Flux", "Source", "Tdfunc", "Ubou", "Fbou", "FbouHdg",
             "Initu", "VisScalars", "VisVectors", "VisTensors",
             "QoIvolume", "QoIboundary"]
    outs = [o for o in order if o in present]
    return "outputs " * join(outs, ", ")
end

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Write a text2code-compatible `pdemodel.txt` for `pde` to `dest_path`.
#
# `pde` is the frontend app struct (already preprocessed, so dimensions such as
# nc/ncu/ncq/nco/ncw are set). The symbolic model functions are resolved from
# Main (or the module named by pde.modelfile), exactly like Gencode.gencode.
#
# Covers the kernel set text2code requires plus the common optional ones: Flux,
# Source, Tdfunc, Ubou, Fbou, FbouHdg, Initu, VisScalars, VisVectors,
# VisTensors, QoIvolume, QoIboundary.
#
# Errors if a required model function (flux, ubou, fbou, fbouhdg, initu) is
# missing, mirroring gencode's hard requirements.
function genpdemodel(pde, dest_path)
    app = pde
    xdg, udg, _, _, wdg, _, _, odg, _, _, uhg, nlg, tau, uinf, param, time = syminit(app)

    if isdefined(Main, Symbol(app.modelfile))
        pdemodel = getfield(Main, Symbol(app.modelfile))
    else
        pdemodel = getfield(Main, Symbol("Main"))
    end

    nc = app.nc
    ncu = app.ncu
    u, q = _split_uq(udg, ncu, nc)

    pr = _t2c_printer()
    blocks = String[]
    present = Set{String}()

    has(fn) = isdefined(pdemodel, Symbol(fn))

    # ----- Flux (required) -----
    if !has("flux")
        error("genpdemodel: pde.flux is not defined")
    end
    f = pdemodel.flux(u, q, wdg, odg, xdg, time, param, uinf)
    push!(blocks, _emit_function("Flux", _ELEM_ARGS, "f", _flatten_F(f), pr))
    push!(present, "Flux")

    # ----- Source (required by text2code; default to zero if absent) -----
    if has("source")
        f = pdemodel.source(u, q, wdg, odg, xdg, time, param, uinf)
        push!(blocks, _emit_function("Source", _ELEM_ARGS, "s", _flatten_F(f), pr))
    else
        push!(blocks, _emit_function("Source", _ELEM_ARGS, "s",
                                     [Sym(0) for _ in 1:ncu], pr))
    end
    push!(present, "Source")

    # ----- Tdfunc (mass; required by text2code; default to ones if absent) -----
    if has("mass")
        f = pdemodel.mass(u, q, wdg, odg, xdg, time, param, uinf)
        push!(blocks, _emit_function("Tdfunc", _ELEM_ARGS, "m", _flatten_F(f), pr))
    else
        push!(blocks, _emit_function("Tdfunc", _ELEM_ARGS, "m",
                                     [Sym(1) for _ in 1:ncu], pr))
    end
    push!(present, "Tdfunc")

    # ----- Ubou (required) -----
    if !has("ubou")
        error("genpdemodel: pde.ubou is not defined")
    end
    f = pdemodel.ubou(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
    push!(blocks, _emit_function("Ubou", _FACE_ARGS, "ub", _flatten_F(f), pr))
    push!(present, "Ubou")

    # ----- Fbou (required) -----
    if !has("fbou")
        error("genpdemodel: pde.fbou is not defined")
    end
    f = pdemodel.fbou(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
    push!(blocks, _emit_function("Fbou", _FACE_ARGS, "fb", _flatten_F(f), pr))
    push!(present, "Fbou")

    # ----- FbouHdg (required for hybrid) -----
    if !has("fbouhdg")
        error("genpdemodel: pde.fbouhdg is not defined")
    end
    f = pdemodel.fbouhdg(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
    push!(blocks, _emit_function("FbouHdg", _FACE_ARGS, "fb", _flatten_F(f), pr))
    push!(present, "FbouHdg")

    # ----- Initu (required) -----
    if !has("initu")
        error("genpdemodel: pde.initu is not defined")
    end
    f = pdemodel.initu(xdg, param, uinf)
    push!(blocks, _emit_function("Initu", _INIT_ARGS, "ui", _flatten_F(f), pr))
    push!(present, "Initu")

    # ----- optional visualization / QoI functions -----
    if has("visscalars")
        f = pdemodel.visscalars(u, q, wdg, odg, xdg, time, param, uinf)
        push!(blocks, _emit_function("VisScalars", _ELEM_ARGS, "s", _flatten_F(f), pr))
        push!(present, "VisScalars")
    end
    if has("visvectors")
        f = pdemodel.visvectors(u, q, wdg, odg, xdg, time, param, uinf)
        push!(blocks, _emit_function("VisVectors", _ELEM_ARGS, "s", _flatten_F(f), pr))
        push!(present, "VisVectors")
    end
    if has("vistensors")
        f = pdemodel.vistensors(u, q, wdg, odg, xdg, time, param, uinf)
        push!(blocks, _emit_function("VisTensors", _ELEM_ARGS, "s", _flatten_F(f), pr))
        push!(present, "VisTensors")
    end
    if has("qoivolume")
        f = pdemodel.qoivolume(u, q, wdg, odg, xdg, time, param, uinf)
        push!(blocks, _emit_function("QoIvolume", _ELEM_ARGS, "s", _flatten_F(f), pr))
        push!(present, "QoIvolume")
    end
    if has("qoiboundary")
        f = pdemodel.qoiboundary(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau)
        push!(blocks, _emit_function("QoIboundary", _FACE_ARGS, "fb", _flatten_F(f), pr))
        push!(present, "QoIboundary")
    end

    # ----- assemble -----
    out = String[]
    append!(out, _header(app))
    push!(out, _outputs_line(present))
    push!(out, "")
    push!(out, join(blocks, "\n"))

    text = join(out, "\n")
    if !endswith(text, "\n")
        text *= "\n"
    end

    open(dest_path, "w") do fid
        write(fid, text)
    end

    return dest_path
end
