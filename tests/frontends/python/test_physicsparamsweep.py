import os
import sys
import tempfile

import numpy

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(repo, "frontends", "Python"))

from exasim.Postprocessing.exasim import (  # noqa: E402
    _normalize_physicsparam_sweep,
    _paramcase_output_dir,
    _write_physicsparam_case,
)


def test_matrix_samples():
    pde = {
        "physicsparam": numpy.array([1.0, 0.0]),
        "physicsparamsweep": numpy.array([[0.5, 0.0], [1.0, 1.0]]),
    }
    cases, has_sweep = _normalize_physicsparam_sweep(pde)
    assert has_sweep
    assert cases.shape == (2, 2)
    assert numpy.allclose(cases[1], [1.0, 1.0])


def test_scalar_parameter_value_list():
    pde = {
        "physicsparam": numpy.array([1.0]),
        "physicsparamsweep": [0.5, 1.0, 2.0],
    }
    cases, has_sweep = _normalize_physicsparam_sweep(pde)
    assert has_sweep
    assert cases.shape == (3, 1)
    assert numpy.allclose(cases[:, 0], [0.5, 1.0, 2.0])


def test_structured_grid_and_case_metadata():
    pde = {
        "datapath": tempfile.mkdtemp(),
        "modelnumber": 0,
        "physicsparam": numpy.array([1.0, 0.0]),
        "physicsparamsweep": {"grid": [[0.5, 1.0], [0.0, 1.0]]},
    }
    cases, has_sweep = _normalize_physicsparam_sweep(pde)
    assert has_sweep
    assert cases.shape == (4, 2)

    outdir = _paramcase_output_dir(pde, 2)
    os.makedirs(outdir, exist_ok=True)
    _write_physicsparam_case(outdir, cases[1])
    saved = numpy.loadtxt(os.path.join(outdir, "physicsparam.txt"))
    assert numpy.allclose(saved, cases[1])
    assert outdir.endswith(os.path.join("dataout", "paramcase_0002"))


def test_bad_dimension_rejected():
    pde = {
        "physicsparam": numpy.array([1.0, 0.0]),
        "physicsparamsweep": numpy.array([[1.0, 0.0, 2.0]]),
    }
    try:
        _normalize_physicsparam_sweep(pde)
    except ValueError as exc:
        assert "2 physics parameters" in str(exc)
    else:
        raise AssertionError("malformed physicsparamsweep was not rejected")


if __name__ == "__main__":
    test_matrix_samples()
    test_scalar_parameter_value_list()
    test_structured_grid_and_case_metadata()
    test_bad_dimension_rejected()
