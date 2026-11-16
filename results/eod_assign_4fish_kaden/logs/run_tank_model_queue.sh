#!/bin/bash
# 2026-11-11: my model in the camera tank with L-barriers, interior 4 x water, N 150.
# 1) the 1000-per-fish clean EODs (70 recordings, --frames), 2) every EOD frame of every third recording (24).
cd /Volumes/Lenca_SSD/github/social_electric_fish
PY=~/opt/anaconda3/envs/electric_fish/bin/python
F=results/eod_assign_4fish_kaden/clean_emitter_set_71/best1000_per_fish_71.csv
for t in $($PY -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('$F').video.unique())))"); do
  OMP_NUM_THREADS=1 $PY kaden_4fish_tank_model.py --rec $t --frames $F --workers 7 2>&1 | grep -v Warning | grep -v "s = np.r_" | tail -1
  mv results/eod_assign_4fish_kaden/$t/model_N150_tank_barriers_int4_by_frame.pkl results/eod_assign_4fish_kaden/$t/model_N150_tank_barriers_int4_best1000.pkl
done
echo "=== best1000 done $(date)"
for t in $(cat results/eod_assign_4fish_kaden/tank_geometry/every_third_24.txt); do
  OMP_NUM_THREADS=1 $PY kaden_4fish_tank_model.py --rec $t --workers 7 2>&1 | grep -v Warning | grep -v "s = np.r_" | grep -v "frames, .* min left"
done
echo "=== all 24 done $(date)"
