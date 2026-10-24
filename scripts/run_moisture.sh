#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
until ! pgrep -f concept_cgraph >/dev/null; do sleep 60; done
echo "$(date -u +%H:%M:%S) GPU free, launching moisture arm" > results/moisture_status.txt
MECH_NAME=moisture MECH_FEATS=4006,3501,2900 FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-jax/bin/python -m graphcast_sae.storms.skill_conv_run >> out/skill_moisture.log 2>&1
echo "$(date -u +%H:%M:%S) exit=$?" >> results/moisture_status.txt
