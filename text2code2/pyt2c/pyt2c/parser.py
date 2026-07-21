"""Parser for the Exasim ``pdemodel.txt`` DSL.

Mirrors ``text2code/text2code/TextParser.hpp``: reads the global declaration keys
(scalars / vectors / jacobian / hessian / batch / outputs / datatype / framework /
codeformat) and the ``function Name(args) ... end`` blocks, recording each
function's body lines, ``output_size`` name+size, and ``matrix`` declarations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# The fixed set of Exasim model functions, in the canonical order text2code uses
# (TextParser.hpp: ParsedSpec::exasimfunctions). Order matters only for the first
# six being mandatory in exasim codeformat.
EXASIM_FUNCTIONS = [
    "Flux", "Source", "Tdfunc", "Ubou", "Fbou", "FbouHdg",
    "Sourcew", "Output", "Monitor", "Initu", "Initq", "Inituq",
    "Initw", "Initv", "Avfield", "Fint", "EoS", "VisScalars",
    "VisVectors", "VisTensors", "QoIvolume", "QoIboundary", "Fext",
]

_FUNC_RE = re.compile(r"^\s*function\s+(\w+)\(([^)]*)\)\s*$")
_MATRIX_DECL_RE = re.compile(r"^\s*matrix\s+(\w+)\((\d+),(\d+)\)\s*;?\s*$")
_OUTPUT_SIZE_RE = re.compile(r"^\s*output_size\((\w+)\)\s*=\s*(\d+)\s*;?\s*$")
_VECTOR_RE = re.compile(r"(\w+)\((\d+)\)")
_KV_RE = re.compile(r"(\w+)\s+(.+)")


@dataclass
class FunctionDef:
    name: str
    args: list[str]
    body: list[str] = field(default_factory=list)
    output: str = ""
    outputsize: int = 0
    matrices: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class Spec:
    scalars: list[str] = field(default_factory=list)
    vectors: dict[str, int] = field(default_factory=dict)
    namevectors: list[str] = field(default_factory=list)
    jacobian: list[str] = field(default_factory=list)
    hessian: list[str] = field(default_factory=list)
    batch: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    datatype: str = "dstype"
    framework: str = "kokkos"
    codeformat: str = "exasim"
    functions: list[FunctionDef] = field(default_factory=list)

    @property
    def exasim(self) -> bool:
        return self.codeformat.lower() == "exasim"

    def function(self, name: str) -> FunctionDef | None:
        for f in self.functions:
            if f.name == name:
                return f
        return None

    def is_output(self, name: str) -> bool:
        return name in self.outputs


def _split(value: str) -> list[str]:
    return [tok.strip() for tok in value.split(",") if tok.strip() != ""]


def parse_string(text: str) -> Spec:
    spec = Spec()
    in_function = False
    current: FunctionDef | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("function"):
            m = _FUNC_RE.match(line)
            if m:
                if in_function and current is not None:
                    spec.functions.append(current)
                current = FunctionDef(name=m.group(1), args=_split(m.group(2)))
                in_function = True
                continue
        if in_function:
            assert current is not None
            m = _MATRIX_DECL_RE.match(line)
            if m:
                current.matrices[m.group(1)] = (int(m.group(2)), int(m.group(3)))
                current.body.append(line)
                continue
            m = _OUTPUT_SIZE_RE.match(line)
            if m:
                current.output = m.group(1)
                current.outputsize = int(m.group(2))
                current.body.append(line)
                continue
            if line == "end":
                spec.functions.append(current)
                current = None
                in_function = False
                continue
            current.body.append(line)
            continue
        # global key = value
        m = _KV_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        toks = _split(value)
        if key == "scalars":
            spec.scalars = toks
        elif key == "vectors":
            for tok in toks:
                vm = _VECTOR_RE.match(tok)
                if vm:
                    spec.vectors[vm.group(1)] = int(vm.group(2))
                    spec.namevectors.append(vm.group(1))
        elif key == "jacobian":
            spec.jacobian = toks
        elif key == "hessian":
            spec.hessian = toks
        elif key == "batch":
            spec.batch = toks
        elif key == "outputs":
            spec.outputs = toks
        elif key == "datatype":
            spec.datatype = value.strip()
        elif key == "framework":
            spec.framework = value.strip()
        elif key == "codeformat":
            spec.codeformat = value.strip()

    if in_function and current is not None:
        spec.functions.append(current)

    if spec.exasim:
        for name in EXASIM_FUNCTIONS[:6]:
            if name not in spec.outputs:
                raise ValueError(
                    f'"{name}" is not listed in `outputs` of the model file. '
                    "Exasim codeformat requires the first six core outputs."
                )
    return spec


def parse_file(path: str) -> Spec:
    with open(path, "r", encoding="utf-8") as f:
        return parse_string(f.read())
