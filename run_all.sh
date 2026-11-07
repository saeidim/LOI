#!/bin/sh
# Regenerates all results, figures and the LaTeX number macros.
set -e
cd "$(dirname "$0")/experiments"
python3 exp_theory.py
python3 exp_main.py
python3 exp_sens.py
python3 exp_robust.py
python3 exp_calib.py
python3 make_figs.py
python3 make_numbers.py ../results/numbers.tex
echo "copy ../figs/*.eps and ../results/numbers.tex into the LaTeX folder"
