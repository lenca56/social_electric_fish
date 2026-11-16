#!/bin/bash
# wait for the 4-fish model voltages, then: 2-fish offset sweep -> full solve at its best offset; 3-fish full solve at -4
cd /Volumes/Lenca_SSD/github/social_electric_fish
PY=~/opt/anaconda3/envs/electric_fish/bin/python
while pgrep -f model_voltages_4fish.py > /dev/null; do sleep 60; done
export OMP_NUM_THREADS=1
$PY -u controls_offset_voltages.py --rec 2-fish --sweep 300 2>&1 | grep -v Warning
BEST=$($PY -c "import pandas as pd; s=pd.read_csv('results/eod_ground_truth/conductivities/offset_sweep_emitter_random300.csv'); m=s.filter(like='d_').median(); print(int(m.idxmin()[2:]))")
echo "2-fish best offset: $BEST"
$PY -u controls_offset_voltages.py --rec 2-fish --offset $BEST 2>&1 | grep -v Warning
$PY -u controls_offset_voltages.py --rec 3-fish --offset -4 2>&1 | grep -v Warning
echo controls finished
