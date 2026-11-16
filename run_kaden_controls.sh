#!/bin/bash
# 2- and 3-fish silent controls with Kaden's masks: every model run (2-fish and 3-fish in parallel at each step), Denis' pipeline on
# the Kaden mask midlines, then the assignment strategies and the accuracy figures. Resumable: each run has its checkpoint.
# Masks QC (kaden_mask_qc_controls.py) and the Kaden midline files (--masks kaden --midlines-only) are made beforehand.
#   ./run_kaden_controls.sh            logs in results/eod_assignment_<n>_fish_silent/with_Kaden_tracking/logs/
cd "$(dirname "$0")"
E=~/opt/anaconda3/envs/electric_fish/bin/python
S=~/opt/anaconda3/envs/ssm/bin/python
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
for n in 2 3; do mkdir -p results/eod_assignment_${n}_fish_silent/with_Kaden_tracking/logs; done
log() { echo "results/eod_assignment_${1}_fish_silent/with_Kaden_tracking/logs/$2.log"; }

both() {   # $1 = log name, rest = runner arguments; 2-fish with 3 workers, 3-fish with 4
    name=$1; shift
    $E eod_assignment_control_groups.py --rec 2-fish --masks kaden "$@" --workers 3 > "$(log 2 $name)" 2>&1 &
    $E eod_assignment_control_groups.py --rec 3-fish --masks kaden "$@" --workers 4 > "$(log 3 $name)" 2>&1 &
    wait; echo "$name done $(date +%H:%M)"; grep -h "% correct" "$(log 2 $name)" "$(log 3 $name)"
}

both N150_square          --N 150 --geometry image --versions none
both N150_square_nobodies --N 150 --geometry image --versions none --bodies no
both N200_tank_nobodies   --N 200 --geometry tank  --versions none --bodies no
both N200_tank            --N 200 --geometry tank  --versions none
both N150_tank_interior   --N 150 --geometry tank  --versions none --interior-sigmas 2,20,40,100

R=/Volumes/Lenca_SSD/github/EODAssignment/analysis/social_electric_fish_runners/denis_kaden_controls_runner.py
$S $R --rec 2-fish > "$(log 2 denis)" 2>&1 & $S $R --rec 3-fish > "$(log 3 denis)" 2>&1 & wait
echo "denis done $(date +%H:%M)"; grep -h "% correct" "$(log 2 denis)" "$(log 3 denis)"

$E assignment_strategies_kaden.py > results/eod_assignment_2_fish_silent/with_Kaden_tracking/logs/strategies.log 2>&1
echo "strategies done $(date +%H:%M)"
