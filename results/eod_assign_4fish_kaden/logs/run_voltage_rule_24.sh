#!/bin/bash
# Voltage rule on each of the 24 recordings, as soon as its full model run is done (tank_model_queue.log).
cd /Volumes/Lenca_SSD/github/social_electric_fish
PY=~/opt/anaconda3/envs/electric_fish/bin/python; LOG=results/eod_assign_4fish_kaden/logs/tank_model_queue.log
for t in $(cat results/eod_assign_4fish_kaden/tank_geometry/every_third_24.txt); do
  until awk -v t="$t" 'f && index($0, t " done:") == 1 {found = 1} /=== best1000 done/ {f = 1} END {exit !found}' $LOG; do sleep 60; done
  OMP_NUM_THREADS=1 $PY kaden_4fish_voltage_rule.py --rec $t 2>&1 | grep -v Warning | tail -2
done
echo "=== voltage rule on all 24 done $(date)"
