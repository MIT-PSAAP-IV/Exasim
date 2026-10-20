#!/usr/bin/env bash
# [nonmatching] Phase 9: run test_interfacetype.jl; exit 77 (skip) without julia.
command -v julia >/dev/null 2>&1 || { echo "SKIP: julia not found"; exit 77; }
exec julia --startup-file=no "$(dirname "$0")/test_interfacetype.jl"
