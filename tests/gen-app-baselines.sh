#!/bin/bash
# Generate golden app baselines from a built Exasim install (run once on main).
# Edit SRC/INSTALL/KOKKOS to point at the baseline tree; outputs to app-baselines/.
set -uo pipefail
P=/Users/teoc/projects/psaap4
set +u; source $P/.stage/condaenv.sh 2>/dev/null || true; set -u
SRC=$P/Exasim-master-verify
INSTALL=$P/Exasim-master-verify-build/install
KOKKOS=$P/Exasim-master-verify-build/deps/kokkos/buildserial
DATA=$INSTALL/share/exasim
BASE=$P/app-baselines
APPS="poisson/poisson2d poisson/poisson3d poisson/periodic poisson/lshape poisson/orion poisson/isoq3d poisson/cone navierstokes/naca0012steady navierstokes/naca0012unsteady navierstokes/nsmach8 navierstokes/sharpb2 navierstokes/isoq"
mkdir -p $BASE
for app in $APPS; do
  name=$(echo $app | tr '/' '_'); adir=$SRC/apps/$app; bdir=/tmp/ab_$name
  [ -f "$adir/grid.bin" ] || { echo "SKIP $app"; continue; }
  cmake -S "$adir" -B "$bdir" -DCMAKE_PREFIX_PATH="$INSTALL;$KOKKOS" >/tmp/cfg_$name.log 2>&1 \
    && cmake --build "$bdir" -j6 >/tmp/bld_$name.log 2>&1 || { echo "BUILD_FAIL $app"; continue; }
  np=$(grep -oE 'mpiprocs *= *[0-9]+' $adir/pdeapp.txt | grep -oE '[0-9]+'); np=${np:-1}
  run=/tmp/run_$name; rm -rf $run; mkdir -p $run; cp $adir/*.bin $adir/*.txt $run/ 2>/dev/null
  (cd $run && EXASIM_DATA_DIR=$DATA mpirun -np $np $bdir/exasimapp pdeapp.txt >/tmp/run_$name.log 2>&1) \
    || { echo "RUN_FAIL $app (np=$np)"; tail -3 /tmp/run_$name.log; continue; }
  out=$BASE/$name; rm -rf $out; mkdir -p $out
  cp $run/dataout/outudg_np*.bin $out/ 2>/dev/null
  cp $run/dataout/outqoi.txt $out/ 2>/dev/null
  echo "OK $app  np=$np  files=$(ls $out|wc -l|tr -d ' ')  size=$(du -sh $out|cut -f1)  qoi=$(head -c 60 $out/outqoi.txt 2>/dev/null|tr '\n' ' ')"
done
echo BASELINE_DONE
