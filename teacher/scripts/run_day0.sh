#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PREP=${PREP:-data/teacher_prep}
CHECK_STEPS=${CHECK_STEPS:-2000}

python download_dataset.py
python -m teacher.verify_third_party
python -m teacher.tests.run_tests test_networks test_losses test_data test_phema test_pilot
python -m teacher.prepare_data --out "$PREP" --floors --device cuda
python -m teacher.benchmark --prep-dir "$PREP" --out runs/benchmark --check-steps "$CHECK_STEPS"
