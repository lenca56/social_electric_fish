#!/bin/bash
# 2- and 3-fish silent controls with Kaden's newer masks (data/<n>_fish_silent/<n>fish_video..._latest_run/, 2026-11-16): mask QC,
# mask midlines, Denis' pipeline on them, my model in its best / cheap versions (no bodies; tank with interior 2), then the assignment
# strategies and the accuracy figures (as run_kaden_controls.sh, which ran every model version on the older Kaden masks).
# Resumable: each model run has its checkpoint.
#   ./run_latest_run_controls.sh       logs in results/eod_assignment_<n>_fish_silent/with_latest_run_tracking/logs/
cd "$(dirname "$0")"
E=~/opt/anaconda3/envs/electric_fish/bin/python
S=~/opt/anaconda3/envs/ssm/bin/python
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
for n in 2 3; do mkdir -p results/eod_assignment_${n}_fish_silent/with_latest_run_tracking/logs; done
log() { echo "results/eod_assignment_${1}_fish_silent/with_latest_run_tracking/logs/$2.log"; }

$E kaden_mask_qc_controls.py --masks latest --workers 7 > "$(log 2 mask_qc)" 2>&1; cp "$(log 2 mask_qc)" "$(log 3 mask_qc)"
echo "mask QC done $(date +%H:%M)"

both() {   # $1 = log name, rest = runner arguments; 2-fish with 3 workers, 3-fish with 4
    name=$1; shift
    $E eod_assignment_control_groups.py --rec 2-fish --masks latest "$@" --workers 3 > "$(log 2 $name)" 2>&1 &
    $E eod_assignment_control_groups.py --rec 3-fish --masks latest "$@" --workers 4 > "$(log 3 $name)" 2>&1 &
    wait; echo "$name done $(date +%H:%M)"; grep -h "% correct\|midlines of" "$(log 2 $name)" "$(log 3 $name)"
}

both midlines --midlines-only

R=/Volumes/Lenca_SSD/github/EODAssignment/analysis/social_electric_fish_runners/denis_kaden_controls_runner.py
$S $R --rec 2-fish --masks latest > "$(log 2 denis)" 2>&1 & $S $R --rec 3-fish --masks latest > "$(log 3 denis)" 2>&1 & wait
echo "denis done $(date +%H:%M)"; grep -h "% correct" "$(log 2 denis)" "$(log 3 denis)"

both N150_square_nobodies --N 150 --geometry image --versions none --bodies no
both N200_tank_nobodies   --N 200 --geometry tank  --versions none --bodies no
both N150_tank_interior   --N 150 --geometry tank  --versions none --interior-sigmas 2

$E assignment_strategies_kaden.py latest > results/eod_assignment_2_fish_silent/with_latest_run_tracking/logs/strategies.log 2>&1
echo "strategies done $(date +%H:%M)"
