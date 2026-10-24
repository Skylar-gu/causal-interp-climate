#!/usr/bin/env bash
# STEP 2 (dose-response) + STEP 3 (bidirectional gain sweep). 2026-08-17.
#
# STEP 2. The library already shows effect monotone in calibrated ascent-loading at
# 28.5s -> 2.79, 13.2s -> 0.32, 6.0s -> 0.22, 2.1s -> -0.03, but with a large gap
# between 28.5 and 13.2 and only five points. Three new groups fill it in, selected
# on ascent sigma ALONE (calibrated label == ascent, firing rate inside the
# convection band 606-9036, disjoint from every group ever run):
#
#   asc21   553, 866,1981   median ascent +20.6s
#   asc17  3357,1033,3314   median ascent +16.8s
#   asc09   141,1538, 591   median ascent  +9.0s
#
# Full TC battery (7 developing + nondev) so the numbers are medians comparable to
# the committed 2.794. ~27 min/arm.
#
# STEP 3. The gain sweep asks the other half of the causal question. delta_gain
# scales the excess above normal: g=0 IS the committed ablation (bit-identical),
# g=1 is baseline, g>1 amplifies. Three storms spanning the observed effect range
# (haishen 7.47 high, ida 2.79 median, patricia 0.20 low) x 6 gains, plus a
# matched-gain RANDOM control at the largest g so amplification has its own null.
#
# Amplification is off-distribution by construction. The deliverable is the CURVE;
# a single amplified gain is not interpretable on its own, and a non-monotone or
# exploding response is the instrument's limit, not the model's physics.
set -u
cd "$(dirname "$0")/.."
export FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
ST=results/step23_status.txt
: > "$ST"

gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
free_count=0; waited=0
while [ "$free_count" -lt 2 ]; do
  U=$(gpu_used)
  if [ "$U" -lt 2000 ]; then free_count=$((free_count + 1)); else
    free_count=0
    [ $((waited % 600)) -eq 0 ] && echo "$(date -u +%H:%M:%S) waiting for GPU (${U} MiB)" >> "$ST"
  fi
  if [ "$free_count" -lt 2 ]; then sleep 30; waited=$((waited + 30)); fi
  if [ "$waited" -gt 28800 ]; then echo "ABORT: GPU busy 8 h" >> "$ST"; exit 1; fi
done
echo "$(date -u +%H:%M:%S) GPU free after ${waited}s, starting" >> "$ST"

# rc must be captured BEFORE any command substitution: `echo "$(date) rc=$?"`
# reports date's status, which is how a dead arm logged rc=0 earlier today.
dose_arm () {
  local name="$1" feats="$2"
  mkdir -p "results/skill/${name}"
  cp -n results/skill/convection/era5_truth.npy "results/skill/${name}/era5_truth.npy"
  echo "$(date -u +%H:%M:%S) === ${name}  ${feats} ===" >> "$ST"
  MECH_NAME="$name" MECH_FEATS="$feats" \
    .venv-jax/bin/python -m graphcast_sae.storms.skill_conv_run > "out/${name}.log" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%S) ${name} rc=${rc}" >> "$ST"
  [ "$rc" -ne 0 ] && { echo "  ${name} FAILED, not scoring" >> "$ST"; return 1; }
  MECH_RES="$name" .venv-jax/bin/python -m graphcast_sae.storms.skill_conv_analyze >> "out/${name}.log" 2>&1
  echo "$(date -u +%H:%M:%S) ${name} scored rc=$?" >> "$ST"
}

dose_arm mech_asc21 553,866,1981
dose_arm mech_asc17 3357,1033,3314
dose_arm mech_asc09 141,1538,591

# ---- STEP 3: gain CURVES for four groups spanning the ascent axis -----------
# One curve per group, so the plot answers whether the amplification response
# scales with calibrated ascent-loading the way the ablation response does.
# Gains are dense on the amplification side (1.25, 1.5, 1.75, 2, 2.5, 3) because
# that is the unexplored half; g=0 is carried in every run as the ablation anchor
# and a regression check against the committed number.
#
#   gain_conv    2401,2067,3174   +28.5s   the committed group
#   gain_asc17   3357,1033,3314   +16.8s
#   gain_asc09    141,1538, 591    +9.0s
#   gain_moist2   2958,2671,  37    +2.1s   near-zero ablation -> expect a flat curve
#
# 3 storms spanning the observed ablation range: haishen (7.47), ida (2.79),
# patricia (0.20). Patricia matters most here -- if amplification moves a storm
# whose ablation was null, the response is asymmetric and that is a finding.
GAINSET=0,1.25,1.5,1.75,2,2.5,3

gain_arm () {
  local name="$1" feats="$2" ov="${3:-0}"
  mkdir -p "results/skill/${name}"
  cp -n results/skill/convection/era5_truth.npy "results/skill/${name}/era5_truth.npy"
  echo "$(date -u +%H:%M:%S) === ${name}  ${feats}  g=${GAINSET} ===" >> "$ST"
  MECH_NAME="$name" MECH_FEATS="$feats" MECH_ALLOW_OVERLAP="$ov" MECH_GAINS="$GAINSET" \
    .venv-jax/bin/python -m graphcast_sae.storms.skill_conv_run \
    haishen2020 ida2021 patricia2015 > "out/${name}.log" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%S) ${name} rc=${rc}" >> "$ST"
}

gain_arm gain_conv   2401,2067,3174 1
gain_arm gain_asc17  3357,1033,3314
gain_arm gain_asc09  141,1538,591
gain_arm gain_moist2 2958,2671,37

echo "$(date -u +%H:%M:%S) ALL DONE" >> "$ST"
