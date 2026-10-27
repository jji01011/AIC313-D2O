#!/bin/bash
# A piece of a teacher-centred line (runs/e112_treal*, 2026-10-15: the discriminator takes samples of the 58.7M teacher
# as its real images), selected on 15,100 other samples of that teacher (data/cache/teacher58_ref, never shown to a
# discriminator; results/reference_teacher58). The teacher was trained on the train split only.
#   bash runs/tref_score.sh <run> <kimg>
cd /home/jiin/AIC313-D2O || exit 1
exec bash runs/ref_score.sh results/reference_teacher58 results/tref_calibration.log 'e112_treal[a-z0-9_]*' "teacher-centred line" "$1" "$2"
