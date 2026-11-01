"""Focused disabled-feature guards; no solver, generated input or installation.

Run: python3 tests/frontends/test_conformal_only_inputs.py
Requires numpy. When sympy is absent, explicitly isolate the production Python
function bodies to verify their early exits without importing symbolic tools.
Octave and Julia validator checks run when those interpreters are available.
"""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy unavailable")
    sys.exit(77)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontends/Python/exasim"
MESSAGE = "not available in this source release"
check = runpy.run_path(str(FRONTEND / "Preprocessing/checkconformalinterface.py"))["checkconformalinterface"]


def rejects(function, *args):
    try:
        function(*args)
    except ValueError as exc:
        assert MESSAGE in str(exc), str(exc)
    else:
        raise AssertionError("unsupported input accepted by " + function.__name__)


paths = {"preprocessing": "Preprocessing/Preprocessing.py",
         "writeapp": "Preprocessing/writeapp.py",
         "exasim": "Postprocessing/exasim.py",
         "exporttext2code": "Gencode/exporttext2code.py"}
functions = {}
if importlib.util.find_spec("sympy") is not None:
    sys.path.insert(0, str(FRONTEND.parent))
    from importlib import import_module
    for name, path in paths.items():
        functions[name] = getattr(import_module("exasim." + path[:-3].replace("/", ".")), name)
    print("Python coverage: real frontend package imports")
else:
    for name, relative in paths.items():
        path = FRONTEND / relative
        node = next(item for item in ast.parse(path.read_text()).body
                    if isinstance(item, ast.FunctionDef) and item.name == name)
        namespace = {"checkconformalinterface": check, "numpy": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        functions[name] = namespace[name]
    print("Python coverage: isolated production function-body early exits only (sympy absent)")

for app in ({}, {"interfacetype": 0}, {"interactiontable": [], "interactions": []}):
    original = copy.deepcopy(app)
    check(app)
    check([app, {}])
    assert app == original

with tempfile.TemporaryDirectory(prefix="exasim-conformal-inputs-") as work:
    output = str(Path(work) / "must-not-be-created")
    for app in ({"interfacetype": 1}, {"interfacetype": -1}, {"interfacetype": 2},
                {"interactiontable": [1]}, {"interactions": [{}]},
                {"interactionfile": "missing-interaction.txt"}):
        rejects(check, app)
        rejects(check, [{}, app])
        rejects(functions["preprocessing"], app, {})
        rejects(functions["writeapp"], app, output)
        rejects(functions["exasim"], app, {})
        rejects(functions["exporttext2code"], app, {}, output)
    rejects(functions["exasim"], [{}, {}], [{}, {}], [{}])
    assert not os.listdir(work), "rejection created output files"

octave = shutil.which("octave")
if octave:
    path = str(ROOT / "frontends/Matlab/Preprocessing").replace("'", "''")
    code = (f"addpath('{path}'); checkconformalinterface(struct()); "
            "checkconformalinterface(struct('interfacetype',0,'interactiontable',[])); "
            "cases={struct('interfacetype',1),struct('interfacetype',2),"
            "struct('interactiontable',1),struct('interactions',struct('type',1)),"
            "struct('interactionfile','missing.txt')}; "
            "for k=1:numel(cases); try; checkconformalinterface(cases{k}); "
            "error('unexpected acceptance'); catch e; "
            "assert(~isempty(strfind(e.message,'not available in this source release'))); end; end;")
    subprocess.run([octave, "--quiet", "--eval", code], check=True, timeout=60)
else:
    print("SKIP: MATLAB-compatible validator execution (Octave unavailable)")

julia = shutil.which("julia")
if julia:
    path = json.dumps(str(ROOT / "frontends/Julia/Exasim/src/Preprocessing/checkconformalinterface.jl"))
    code = (f"include({path}); checkconformalinterface(Dict()); "
            "checkconformalinterface((interfacetype=0,interactiontable=[],interactions=[])); "
            "for app in [Dict(:interfacetype=>1),Dict(:interfacetype=>2),"
            "Dict(:interactiontable=>[1]),Dict(:interactions=>[1]),"
            "Dict(:interactionfile=>\"missing.txt\")]; "
            "try; checkconformalinterface(app); error(\"unexpected acceptance\"); "
            "catch e; occursin(\"not available in this source release\",sprint(showerror,e)) || rethrow(); end; end")
    subprocess.run([julia, "--startup-file=no", "-e", code], check=True, timeout=60)
else:
    print("SKIP: Julia validator execution (Julia unavailable)")

print("PASS: conformal controls and early nonmatching/generalized-input rejection")
