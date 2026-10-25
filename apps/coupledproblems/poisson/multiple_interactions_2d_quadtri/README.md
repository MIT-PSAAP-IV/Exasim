# multiple_interactions_2d_quadtri

Four Poisson slabs with alternating quadrilateral and triangular meshes are
coupled in a chain by three type-1 non-matching interactions at `x = 0.25`,
`0.50`, and `0.75`. Always configure and run an absolute temporary copy; never
run from this source directory:

```bash
export EXASIM_ROOT=/absolute/path/to/Exasim
export EXASIM_PREFIX=/absolute/path/to/current/install
case_dir=$(mktemp -d /tmp/multiple_interactions_2d_quadtri.XXXXXX)
cp -R "$EXASIM_ROOT/apps/coupledproblems/poisson/multiple_interactions_2d_quadtri/." "$case_dir/"
cmake -S "$case_dir" -B "$case_dir/build" -DCMAKE_PREFIX_PATH="$EXASIM_PREFIX" -DEXASIM_TEXT2CODE_EXECUTABLE="$EXASIM_PREFIX/bin/text2code"
cmake --build "$case_dir/build" --parallel
cd "$case_dir"
mpirun -np 4 "$case_dir/build/exasimapp" --interactions "$case_dir/interaction.txt" \
  "$case_dir/pdeapp1.txt" 1 "$case_dir/pdeapp2.txt" 1 \
  "$case_dir/pdeapp3.txt" 1 "$case_dir/pdeapp4.txt" 1
```

For the validated unequal partition, use six MPI ranks and assign `1 2 2 1`
ranks to the four domains:

```bash
mpirun -np 6 "$case_dir/build/exasimapp" --interactions "$case_dir/interaction.txt" \
  "$case_dir/pdeapp1.txt" 1 "$case_dir/pdeapp2.txt" 2 \
  "$case_dir/pdeapp3.txt" 2 "$case_dir/pdeapp4.txt" 1
```

Both layouts converge to a final linear residual below `9e-15`. Their assembled
solutions agree to about `8e-15` in relative norm on the current mesh. Do not
use the formerly documented `1 2 3 1` layout with this small case: partitioning
domain 3 over three ranks leaves an endpoint rank with no elements, so interface
setup correctly rejects the layout. Refine the mesh before assigning more ranks
to that domain.

Generated `datain*`, `dataout*`, builds, and logs must remain outside the source
checkout.
