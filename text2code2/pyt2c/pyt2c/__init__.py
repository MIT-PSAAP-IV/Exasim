"""pyt2c — a pip-symengine reimplementation of Exasim's text2code model codegen.

Parses a ``pdemodel.txt`` and emits a header-only concrete model
(``generated/my_model.hpp``) equivalent to the C++ ``text2code`` output, in a
single stage (no compile-a-program-then-run).
"""
from .parser import parse_file, parse_string, Spec
from .interp import ModelEvaluator
from .codegen import generate_header

__all__ = ["parse_file", "parse_string", "Spec", "ModelEvaluator", "generate_header"]
__version__ = "0.1.0"
