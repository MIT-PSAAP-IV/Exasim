"""Exercise actual Python writers and, optionally, the C++ text/binary bridge."""
import argparse
import copy
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Load the actual writer modules without importing unrelated mesh-generation
# dependencies from the package initializer. This test needs NumPy and SymPy.
for suffix in ("", ".Preprocessing", ".Gencode"):
    name = "_exasim_composite_test" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / "frontends/Python/exasim" / suffix.lstrip("."))]
    sys.modules[name] = module
initializepde = importlib.import_module("_exasim_composite_test.Preprocessing.initializepde").initializepde
writeapp = importlib.import_module("_exasim_composite_test.Preprocessing.writeapp").writeapp
write_text = importlib.import_module("_exasim_composite_test.Gencode.exporttext2code")._write_pdeapp


def contents(path):
    data = np.fromfile(path, dtype=np.float64)
    nsize = int(data[0])
    sizes = data[1:1 + nsize].astype(int)
    payload = data[1 + nsize:]
    dims = payload[:sizes[0]]
    problem_start = sizes[0] + sizes[1]
    return dims, payload[problem_start:problem_start + sizes[2]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", type=Path)
    args = parser.parse_args()
    base = initializepde("")
    assert base["compositehdg"] == 0 and base["compositebatchsize"] == 1024
    mesh = {"boundarycondition": np.array([1]), "boundaryexpr": ["x1 < 0"]}
    with tempfile.TemporaryDirectory(prefix="exasim-composite-python-") as directory:
        temp = Path(directory)
        for cap in (1, 127, 128, 256, 512, 1024, 2048, 4096):
            app = copy.deepcopy(base)
            app.update(compositehdg=1, compositebatchsize=cap, interfacetype=0,
                       uniformrefinementlevel=2, physicsparam=np.array([1.]), tau=np.array([1.]))
            binary = temp / "app.bin"
            writeapp(copy.deepcopy(app), str(binary))
            dims, problem = contents(binary)
            assert tuple(dims[23:25]) == (1, cap)
            assert dims[22] == 0 and tuple(problem[33:35]) == (2, 0)
            text = temp / "pdeapp.txt"
            write_text(app, mesh, {"meshfile": "mesh.bin"}, str(text))
            assert "compositehdg = 1;" in text.read_text()
            assert f"compositebatchsize = {cap};" in text.read_text()
            if args.converter:
                subprocess.run([str(args.converter.resolve()), "--convert", str(text), str(binary)], check=True)
                dims, problem = contents(binary)
                assert tuple(dims[23:25]) == (1, cap)
                assert problem[33] == 2
        for codes, expected in [(5, [5]), ([1001, 5, 5], [5, 1001]), ([], [])]:
            app = copy.deepcopy(base)
            app.update(compositehdg=1, compositebc=codes, interfacetype=0,
                       physicsparam=np.array([1.]), tau=np.array([1.]))
            writeapp(copy.deepcopy(app), str(binary))
            dims, _ = contents(binary)
            assert dims[25] == len(expected) and list(dims[40:]) == expected
            write_text(app, mesh, {"meshfile": "mesh.bin"}, str(text))
            if args.converter:
                subprocess.run([str(args.converter.resolve()), "--convert", str(text), str(binary)], check=True)
                dims, _ = contents(binary)
                assert dims[25] == len(expected) and list(dims[40:]) == expected
        unsupported = copy.deepcopy(base)
        unsupported.update(compositehdg=1, interfacetype=1)
        try:
            writeapp(unsupported, str(temp / "nonmatching.bin"))
        except ValueError as exc:
            assert "not available in this source release" in str(exc)
        else:
            raise AssertionError("nonmatching input accepted")
        invalid = {
            "compositebc": (0, -1, 5.5, np.nan, np.inf, 2147483648, 5j, "5", [5, np.nan], [[5]], True),
            "compositehdg": (-1, 2, 0.5, np.nan, np.inf, 1j, [1]),
            "compositebatchsize": (0, -1, 1024.5, np.nan, np.inf, -np.inf,
                                   2147483648, 1024j, [1024], "1024"),
        }
        for key, values in invalid.items():
            for value in values:
                app = copy.deepcopy(base)
                app[key] = value
                try:
                    writeapp(app, str(temp / "invalid.bin"))
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"accepted {key}={value!r}")
        app = copy.deepcopy(base)
        del app["compositehdg"], app["compositebatchsize"], app["compositebc"]
        writeapp(app, str(temp / "legacy.bin"))
        dims, _ = contents(temp / "legacy.bin")
        assert tuple(dims[23:25]) == (0, 1024)
    print("COMPOSITE_PYTHON_FRONTEND_PASS")


if __name__ == "__main__":
    main()
