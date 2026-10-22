# multiple_interactions_3d_hextet

Four Poisson slabs with alternating hexahedral and tetrahedral meshes are
coupled in a hex--tet--hex--tet chain by three type-1 non-matching interactions
at `x = 0.25`, `0.50`, and `0.75`. Always configure and run an absolute
temporary copy; never run from this source directory:

```bash
export EXASIM_ROOT=/absolute/path/to/Exasim
export EXASIM_PREFIX=/absolute/path/to/current/install
case_dir=$(mktemp -d /tmp/multiple_interactions_3d_hextet.XXXXXX)
cp -R "$EXASIM_ROOT/apps/coupledproblems/poisson/multiple_interactions_3d_hextet/." "$case_dir/"
cmake -S "$case_dir" -B "$case_dir/build" -DCMAKE_PREFIX_PATH="$EXASIM_PREFIX" -DEXASIM_TEXT2CODE_EXECUTABLE="$EXASIM_PREFIX/bin/text2code"
cmake --build "$case_dir/build" --parallel
cd "$case_dir"
mpirun -np 4 "$case_dir/build/exasimapp" --interactions "$case_dir/interaction.txt" \
  "$case_dir/pdeapp1.txt" 1 "$case_dir/pdeapp2.txt" 1 \
  "$case_dir/pdeapp3.txt" 1 "$case_dir/pdeapp4.txt" 1
```

Two unequal layouts have been validated:

```bash
# Six ranks: one, two, two, and one rank per domain.
mpirun -np 6 "$case_dir/build/exasimapp" --interactions "$case_dir/interaction.txt" \
  "$case_dir/pdeapp1.txt" 1 "$case_dir/pdeapp2.txt" 2 \
  "$case_dir/pdeapp3.txt" 2 "$case_dir/pdeapp4.txt" 1

# Seven ranks: one, two, three, and one rank per domain.
mpirun -np 7 "$case_dir/build/exasimapp" --interactions "$case_dir/interaction.txt" \
  "$case_dir/pdeapp1.txt" 1 "$case_dir/pdeapp2.txt" 2 \
  "$case_dir/pdeapp3.txt" 3 "$case_dir/pdeapp4.txt" 1
```

All three layouts converge to a final linear residual below `7e-13`. The two
unequal-layout solutions agree with the equal-layout solution to about `2e-12`
in relative norm on the current mesh. Each run reports all six directed donor
searches with zero points outside the donor face.

The nested, tetrahedral-owner direction alternates along the chain: endpoint 2
for domains 1--2, endpoint 1 for domains 2--3, and endpoint 2 for domains 3--4.
Those three directions have zero nesting violations. The opposite,
hexahedral-owner directions can cross several tetrahedral donor faces; their
reported mixed-nesting count is therefore diagnostic and is not a failure of
the coupling.

Generated `datain*`, `dataout*`, builds, and logs must remain outside the source
checkout.
