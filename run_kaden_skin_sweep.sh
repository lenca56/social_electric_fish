#!/bin/bash
# N150 tank, other fish with interior 2 x water and the skin as resistive as l cm of water, l = 0 (no skin), 0.03, 0.3, 1, 3
# (l = 0.1 is the run N150_tank_int2_none), Kaden masks, 2- and 3-fish in parallel. Logs in with_Kaden_tracking/logs/.
cd "$(dirname "$0")"; E=~/opt/anaconda3/envs/electric_fish/bin/python; export OMP_NUM_THREADS=1
for n in 2 3; do
  w=$([ $n = 2 ] && echo 3 || echo 4)
  $E eod_assignment_control_groups.py --rec $n-fish --masks kaden --N 150 --geometry tank --versions none --interior-sigmas 2 \
     --skin-lambdas 0,0.03,0.3,1,3 --workers $w > results/eod_assignment_${n}_fish_silent/with_Kaden_tracking/logs/N150_tank_int2_skin.log 2>&1 &
done
wait; grep -h "% correct" results/eod_assignment_{2,3}_fish_silent/with_Kaden_tracking/logs/N150_tank_int2_skin.log
