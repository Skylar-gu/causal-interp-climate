#!/usr/bin/env bash
# Serialize the convection-skill run behind the other GPU jobs.
# Wait until the known ahead-of-us jobs have EXITED and the card is genuinely free,
# then run the per-storm pipeline. Two flagship bf16 graphs would OOM, so we never
# overlap: we require the queue drained AND GPU<6000 MiB sustained 40s.
set -u
cd "$(dirname "$0")/.."
LOG=out/skill_conv_run.log
echo "[$(date)] launcher up; waiting for GPU queue to drain" | tee -a "$LOG"

ahead_alive() {
  # any of: skill extraction, counterfactual loop/run, 8->9 test, diurnal — plus any OTHER venv-jax python
  pgrep -f "skill_extract_init"        >/dev/null && return 0
  pgrep -f "steer_ida_counterfactual"  >/dev/null && return 0
  pgrep -f "run_mini_8to9"             >/dev/null && return 0
  pgrep -f "mini_8to9_diurnal"         >/dev/null && return 0
  pgrep -f "check_mechanisms_diurnal"  >/dev/null && return 0
  # any counterfactual bash waiter still looping
  pgrep -f "COUNTERFACTUAL_DONE"       >/dev/null && return 0
  return 1
}

while true; do
  if ahead_alive; then
    sleep 60; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${used:-99999}" -lt 6000 ]; then
    sleep 40
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "${used2:-99999}" -lt 6000 ] && ! ahead_alive; then
      echo "[$(date)] GPU free (${used2} MiB), queue drained -> starting run" | tee -a "$LOG"
      FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv-jax/bin/python -m graphcast_sae.storms.skill_conv_run >> "$LOG" 2>&1
      echo "[$(date)] SKILL_CONV_RUN_DONE (exit $?)" | tee -a "$LOG"
      break
    fi
  fi
  sleep 60
done
