#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
exec bash runs/ref_score.sh results/reference_teacher58 results/tref_calibration.log 'e112_treal[a-z0-9_]*' "teacher-centred line" "$1" "$2"
