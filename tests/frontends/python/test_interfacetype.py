"""[nonmatching] Phase 9: interfacetype field, validation and app.bin slot 34 (Python frontend).

Run: python3 tests/frontends/python/test_interfacetype.py   (exit 0 pass, 1 fail, 77 skip)

The modules are loaded from their files so this test needs only numpy (the
package __init__ imports sympy, which is not required for serialization).
Functions that live in sympy-importing modules (the parameter-sweep template
and the text2code value formatter) are extracted from their source with ast,
so the production code itself is exercised, not a copy.
"""
import ast
import copy
import importlib.util
import os
import struct
import sys
import tempfile

try:
    import numpy
except ImportError:
    print("SKIP: numpy unavailable")
    sys.exit(77)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY = os.path.join(ROOT, "frontends", "Python", "exasim")
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print("FAIL:", message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_functions(path, names, namespace):
    tree = ast.parse(open(path, encoding="utf-8").read())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), f"functions {names} not found in {path}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def read_problem(filename):
    """Read app.bin directly: numsize, nsize[numsize], ndims[nsize0], flag[nsize1], problem[nsize2]."""
    with open(filename, "rb") as f:
        data = f.read()
    values = struct.unpack(f"<{len(data) // 8}d", data)
    numsize = int(values[0])
    nsize = [int(v) for v in values[1:1 + numsize]]
    start = 1 + numsize + nsize[0]
    flag = list(values[start:start + nsize[1]])
    start += nsize[1]
    problem = list(values[start:start + nsize[2]])
    return flag, problem


init = load_module("p9_initializepde", os.path.join(PY, "Preprocessing", "initializepde.py"))
writeapp_module = load_module("p9_writeapp", os.path.join(PY, "Preprocessing", "writeapp.py"))
initapp = load_module("p9_initializeapp", os.path.join(PY, "Preprocessing", "initializeapp.py"))


def base_pde():
    pde = init.initializepde("Backend")
    for key in ["nd", "ncx", "nc", "ncu", "ncq", "ncw", "nco", "nch", "ncs", "ncv", "ncp", "nce",
                "npe", "npf", "nfe", "ne", "np", "mpiprocs", "porder", "pgauss", "elemtype",
                "nodetype", "neb", "nfb"]:
        pde.setdefault(key, 1)
    return pde


def write(pde, tmp, name):
    filename = os.path.join(tmp, name)
    app = writeapp_module.writeapp(copy.deepcopy(pde), filename)
    return app, read_problem(filename)


with tempfile.TemporaryDirectory() as tmp:
    # Default, explicit 0 and explicit 1.
    pde = base_pde()
    check(pde.get("interfacetype") == 0, "initializepde default interfacetype is 0")
    _, (_, problem) = write(pde, tmp, "default.bin")
    check(len(problem) >= 35 and problem[34] == 0.0, "omitted/default interfacetype writes 0 at slot 34")
    for value in (0, 1, numpy.int64(1), 1.0):
        pde = base_pde()
        pde["interfacetype"] = value
        _, (_, problem) = write(pde, tmp, "explicit.bin")
        check(problem[34] == float(value), f"interfacetype={value!r} written at slot 34")

    # Lower-level public initializer (Revision 1, N2).
    app = initapp.initializeapp("Backend")
    check(app.get("interfacetype") == 0, "initializeapp default interfacetype is 0")
    pde = base_pde()
    pde["interfacetype"] = app["interfacetype"]
    _, (_, problem) = write(pde, tmp, "initializeapp.bin")
    check(problem[34] == 0.0, "initializeapp default writes 0 at slot 34")

    # A dictionary without the key (older caller-created dictionaries) still writes 0.
    pde = base_pde()
    del pde["interfacetype"]
    _, (_, problem) = write(pde, tmp, "missing.bin")
    check(problem[34] == 0.0, "absent interfacetype key writes 0 at slot 34")

    # Invalid values are rejected before anything is written.
    for value in (2, -1, 0.5, True, [0, 1], "1", numpy.nan, 1 + 0j):
        pde = base_pde()
        pde["interfacetype"] = value
        filename = os.path.join(tmp, "invalid.bin")
        if os.path.exists(filename):
            os.remove(filename)
        try:
            writeapp_module.writeapp(copy.deepcopy(pde), filename)
            check(False, f"interfacetype={value!r} is rejected")
        except ValueError as error:
            check("interfacetype" in str(error), f"interfacetype={value!r} error names the field")
            check(not os.path.exists(filename), f"interfacetype={value!r} writes no app.bin")

    # User extras start at slot 35 and keep their order.
    pde = base_pde()
    pde["interfacetype"] = 1
    pde["problem"] = numpy.array([7, 8, 9])
    _, (_, problem) = write(pde, tmp, "extras.bin")
    check(len(problem) == 38, "35 fixed slots plus three user extras")
    check(problem[34] == 1.0 and problem[35:] == [7.0, 8.0, 9.0], "user extras follow slot 34 in order")

    # Parameter-sweep template round trip: stripping the fixed prefixes of a written
    # app and writing it again must reproduce the same flag/problem/factor/solversparam.
    namespace = {"numpy": numpy, "pycopy": copy}
    extract_functions(os.path.join(PY, "Postprocessing", "exasim.py"), ["_writeapp_template"], namespace)
    pde = base_pde()
    pde["interfacetype"] = 0
    pde["uniformrefinementlevel"] = 1
    pde["problem"] = numpy.array([7, 8])
    pde["flag"] = numpy.array([5])
    first, (flag1, problem1) = write(pde, tmp, "sweep1.bin")
    template = namespace["_writeapp_template"](first)
    second, (flag2, problem2) = write(template, tmp, "sweep2.bin")
    check(problem2 == problem1, "sweep template round trip preserves the problem vector "
          f"(first {len(problem1)} values, second {len(problem2)}; slot 34 {problem1[34]} -> {problem2[34]})")
    check(flag2 == flag1, f"sweep template round trip preserves the flag vector ({len(flag1)} -> {len(flag2)})")
    check(numpy.array_equal(numpy.ravel(second["factor"]), numpy.ravel(first["factor"])),
          "sweep template round trip preserves factor")
    check(numpy.array_equal(numpy.ravel(second["solversparam"]), numpy.ravel(first["solversparam"])),
          "sweep template round trip preserves solversparam")

    # text2code export: the key is exported next to interfacefluxmap, formatted as an integer.
    exporter = os.path.join(PY, "Gencode", "exporttext2code.py")
    source = open(exporter, encoding="utf-8").read()
    check('"interfacefluxmap", "interfacetype"' in source, "text2code exporter lists interfacetype")
    fmt = extract_functions(exporter, ["_format_value", "_format_float"], {"np": numpy})
    for value, text in ((0, "0"), (1, "1"), (numpy.int64(1), "1")):
        check(fmt["_format_value"](value, "interfacetype") == text,
              f"exporter formats interfacetype={value!r} as {text}")

if FAILURES:
    print(f"FAIL: test_interfacetype.py ({len(FAILURES)} failures)")
    sys.exit(1)
print("PASS: test_interfacetype.py")
