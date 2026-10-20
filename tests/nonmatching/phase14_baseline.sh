#!/bin/sh
# Phase 14 T0 reproducibility and tree-neutrality driver.
# Usage: phase14_baseline.sh REF_EXE CAND_EXE OUTPUT_DIR

set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 REF_EXE CAND_EXE OUTPUT_DIR" >&2
    exit 2
fi

reference_exe=$1
candidate_exe=$2
output_root=$3

if [ ! -x "$reference_exe" ] || [ ! -x "$candidate_exe" ]; then
    echo "both baseline emitters must exist and be executable" >&2
    exit 2
fi
if [ -e "$output_root" ]; then
    echo "output directory already exists: $output_root" >&2
    exit 2
fi

mkdir -p "$output_root/environment"
{
    date -u
    uname -a
    cmake --version
    mpiexec --version
    c++ --version
    ulimit -a
} > "$output_root/environment/toolchain.txt" 2>&1

run_tree()
{
    tree=$1
    executable=$2
    repeat=1
    while [ "$repeat" -le 3 ]; do
        for case_spec in 2:1 4:1 4:2 4:3 8:3; do
            np=${case_spec%:*}
            split=${case_spec#*:}
            destination="$output_root/$tree/repeat$repeat/np${np}_split${split}"
            mkdir -p "$destination"
            if [ "$repeat" -eq 1 ]; then
                output_mode=--dump
            else
                output_mode=--manifest
            fi
            mpiexec -n "$np" "$executable" --split "$split" "$output_mode" "$destination" \
              > "$destination/stdout.txt" 2> "$destination/stderr.txt"
        done
        repeat=$((repeat + 1))
    done
}

compare_repeats()
{
    tree=$1
    for case_spec in 2:1 4:1 4:2 4:3 8:3; do
        np=${case_spec%:*}
        split=${case_spec#*:}
        rank=0
        while [ "$rank" -lt "$np" ]; do
            first="$output_root/$tree/repeat1/np${np}_split${split}/manifest_rank${rank}.txt"
            cmp "$first" "$output_root/$tree/repeat2/np${np}_split${split}/manifest_rank${rank}.txt"
            cmp "$first" "$output_root/$tree/repeat3/np${np}_split${split}/manifest_rank${rank}.txt"
            rank=$((rank + 1))
        done
    done
}

compare_trees()
{
    for case_spec in 2:1 4:1 4:2 4:3 8:3; do
        np=${case_spec%:*}
        split=${case_spec#*:}
        rank=0
        while [ "$rank" -lt "$np" ]; do
            cmp "$output_root/reference/repeat1/np${np}_split${split}/manifest_rank${rank}.txt" \
                "$output_root/candidate/repeat1/np${np}_split${split}/manifest_rank${rank}.txt"
            rank=$((rank + 1))
        done
    done
}

run_tree reference "$reference_exe"
run_tree candidate "$candidate_exe"
compare_repeats reference
compare_repeats candidate
compare_trees

find "$output_root" -type f ! -name SHA256SUMS -exec shasum -a 256 {} \; \
  > "$output_root/SHA256SUMS"
echo "PASS Phase 14 T0 byte baselines are deterministic and tree-neutral"
