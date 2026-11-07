# Look or Imagine: code for "World-Model Imagination Versus Beam Sweeping"

This repository regenerates every number, table and figure of the paper
*World-Model Imagination Versus Beam Sweeping: Exact Age-of-Information
Analysis and Conformal Guarantees for mmWave V2I Scheduling*
(S. Iranmanesh and R. Raad).

## Requirements

Python 3.10 or newer with `numpy`, `scipy`, `matplotlib` and `joblib`
(`pip install -r requirements.txt`). `pdflatex` and `pdftops` (poppler) are
needed only to rebuild the system-model figure (Fig. 1) from its TikZ source.

## Reproducing the paper

```
cd experiments
sh ../run_all.sh            # or run the steps below one by one
```

| Step | Script | Output | Paper |
|------|--------|--------|-------|
| 1 | `exp_theory.py` | `results/theory.pkl` | Fig. 2, Table II |
| 2 | `exp_main.py`   | `results/main.pkl`   | Figs. 3, 4, 6, Table III |
| 3 | `exp_sens.py`   | `results/sens.pkl`   | Fig. 5 |
| 4 | `exp_robust.py` | `results/robust.pkl` | Table IV |
| 5 | `make_figs.py`  | `figs/*.eps` (and PDF twins) | all figures |
| 6 | `make_numbers.py ../paper/numbers.tex` | LaTeX macros | every number in the text |

`LOI_JOBS` sets the number of parallel workers (default: all cores). On two
cores the complete pipeline takes about 1.5 hours; the main study alone about
25 minutes. Ground-truth traffic realizations are cached in `results/cache`.

## Package layout (`loi/`)

| Module | Content |
|--------|---------|
| `config.py` | all scenario and algorithm parameters (Table I) |
| `traffic.py` | IDM car following, OU driver noise, overtaking and random lane changes on a 4-lane ring road |
| `channel.py` | ULA gain, TR 38.901 path loss, ITU-R P.526 knife-edge blockage (roof and side), wideband Rician/Rayleigh success table |
| `truth.py` | one traffic realization with common random numbers (trajectories, shadowing, blockage, per-slot fading thresholds, sweep noise) |
| `worldmodel.py` | CWNA Kalman beliefs, learned intensities, physics renderer with sigma-point blockage, link belief with censoring and soft NACK updates, Gauss-Hermite success prediction |
| `aoi.py` | closed form of Theorem 3, exact dynamic programming, Monte Carlo, certificate of Corollary 2 |
| `sim.py` | slot-level simulator and the policies: periodic ZOH, periodic CV, LOI, LOI-R, randomized, oracle |

## Policies

* `zoh`: conventional beam management, beams and link beliefs held between sweeps.
* `cv`: world-model imagination with a fixed look period.
* `loi`: Algorithm 1 (adaptive look period from the exact freshness certificate).
* `loir`: LOI with Clopper-Pearson ACK calibration and conformal beam errors.
* `rand`: periodic CV with the uniform stationary randomized scheduler (Theorem 3(ii)).
* `oracle`: genie beams and channel realizations, no looks.

## Determinism

All randomness is seeded. Traffic, shadowing, fading and sweep noise are drawn
in `truth.py` independently of the policy, so every policy is evaluated on
exactly the same channel realizations and results are paired.
