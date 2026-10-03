# Code for "What a Learned Simulator Gets Wrong About Interventions, and What One Physical Constraint Fixes"

Debojit Biswas — dbiswas2015@fau.edu

Everything in the paper is reproducible from these scripts on a CPU. All data for Experiments 1–5 and the theory are synthetic; Experiments 6–7 need the NGSIM US-101 CSV (see their rows) and generated
by the scripts themselves; Experiments 6–7 need the NGSIM US-101 CSV (data.transportation.gov dataset 8ect-6jqj); everything else is synthetic.

## Requirements
Python ≥ 3.10, `numpy`, `scipy`, `sympy`, `torch` (CPU build is fine), `matplotlib`.

    pip install sympy numpy scipy torch matplotlib

## Map from paper to script

| Paper item | Script | Output | Runtime (2 CPU cores) |
|---|---|---|---|
| Experiment 7, Table 10 (head ablation: parametric/free policy × parametric/free dynamics) | `exp8_ablation.py` (`EXP2_SEED` 11, 21) | `exp8_results.json` | ~5 min per cell |
| Prop. 2, Table 3 now includes the Nickell term (K_wv) | `theory_prop3.py` | `theory_prop3.json` | seconds |
| Quasi-differenced GMM (Hatanaka-type) estimator at every ρ (Sec. 5.1 text) | `theory_iv_qd.py` | `theory_iv_qd.json` | ~2 min |
| Fisher information vs excitation level σ_n (Sec. 5.3 text) | `theory_fisher_sign.py` | `theory_fisher_sign.json` | ~10 min |
| Vector-state Lemma 1, Table 2 (two-state world) | `theory_lemma1_vector.py` | `theory_lemma1_vector.json` | ~1 min |
| Semi-synthetic NGSIM world, Appendix B | `exp7_semisynthetic.py` (`EXP7_RHO`, `EXP7_LAMBDAS`, `EXP2_SEED` env vars; needs the NGSIM slim CSV) | `exp7_results_rho0.80.json`, `exp7_results_rho0.98.json` | ~10 min per model |
| Lemma 1, Table 1 (exact finite-horizon bias; stationary closed form) | `theory_lemma1.py` | stdout table | seconds |
| Props. 1–2 linear estimators (OLS, FE, KB, KB-FE, KB-FD), N = 20 000 | `theory_estimators.py` | `theory_estimators.json` | ~2 min |
| Prop. 2, Table 3 (population limit of KB-FE, hump at ρ* = 0.924 at this parameter point) | `theory_prop3.py` | `theory_prop3.json` | seconds |
| Prop. 3 (Jacobian of the observational law; singular values) | `theory_identifiability.py` | `theory_identifiability.json` | seconds |
| Result 1, Table 4 (Kalman maximum likelihood, standard errors) | `theory_kalman_ml.py` | `theory_kalman_ml.json` | ~25 min |
| Experiment 1, Fig. 2 (neural one-step learners) | `exp1_confounded_vehicle.py`, `plot_exp1.py` | `exp1_results.json`, `exp1_figure.png` | ~50 min |
| Experiment 2, Fig. 3, Table 5 (latent-process NDO, 3 seeds, OOD) | `exp2_latent_process_ndo.py`, `plot_exp2.py` | `exp2_results_seed{S}.json`, `exp2_aggregate.json`, `exp2_figure.png` | ~55 min per seed |
| Experiment 3, Table 6 (persistence sweep, 3 seeds; nonlinear drag, 2 seeds) | `exp3_nonlinear_and_rho.py` | `exp3_results.json`, `exp3_rho_aggregate.json` | ~15 min per cell |
| Experiment 4, Table 7 (grey-box state channel P2; wrong-gain sensitivity) | `exp4_p2_and_sensitivity.py` | `exp4_results.json` | ~8 min per cell |
| Prop. 3 (generic local identifiability, exact rational arithmetic) | `theory_generic_identifiability.py` | stdout | ~1 min |
| Fig. 1 (Fisher information vs persistence) | `theory_fisher_curve.py`, `plot_fisher.py` | `theory_fisher_curve.json`, `fisher_figure.png` | ~5 min |
| Anderson–Hsiao IV baseline (Sec. 5.1) | `theory_iv_baseline.py` | `theory_iv_baseline.json` | ~1 min |
| Experiment 5, Table 8 (incl. DKF-2-KIN) (two-state system, exact kinematic row) | `exp5_twostate.py` | `exp5_results.json` | ~20 min per seed |
| Experiment 6 — NGSIM US-101 car-following transfer test (Table 9; NAIVE, DKF, DKF-P2, DKF-P12, 3 seeds) | `realdata_ngsim.py` | `realdata_results.json` | data: data.transportation.gov dataset 8ect-6jqj, `location=us-101`, columns vehicle_id, frame_id, lane_id, v_vel, v_acc, preceding, space_headway |

## Reproducing

    python3 theory_lemma1.py
    python3 theory_estimators.py
    python3 theory_prop3.py
    python3 theory_identifiability.py
    python3 theory_kalman_ml.py
    python3 exp1_confounded_vehicle.py && python3 plot_exp1.py
    for S in 11 21 22; do EXP2_SEED=$S python3 exp2_latent_process_ndo.py AR1 PWC SINE; done
    python3 aggregate_exp2.py && python3 plot_exp2.py
    python3 exp3_nonlinear_and_rho.py NL
    EXP2_SEED=21 python3 exp3_nonlinear_and_rho.py NL
    for S in 11 21 22; do for R in 0.99 0.7 0.3; do EXP2_SEED=$S python3 exp3_nonlinear_and_rho.py RHO $R; done; done
    for W in AR1 NL RHO0.3; do python3 exp4_p2_and_sensitivity.py P2 $W; done
    for B in 0.8 0.9 1.1 1.2; do python3 exp4_p2_and_sensitivity.py WRONGB $B; done
    python3 theory_generic_identifiability.py
    python3 theory_fisher_curve.py && python3 plot_fisher.py
    python3 theory_iv_baseline.py
    for S in 11 21; do EXP2_SEED=$S python3 exp5_twostate.py; done
    python3 realdata_ngsim.py ngsim_us101_slim.csv --seed 0   # seeds 0,1,2; --synthetic for a dry run without data

Seed 11 of Experiment 2 writes `exp2_results.json` (legacy name) — `aggregate_exp2.py` handles both names.
`EXP2_THREADS` limits torch threads. Set `EXP2_SEED` to change the data/initialisation seed.

## Model and parameters (shared by every script)
Dynamics `v' = v + Δ(b u − k v + g c) + w`, policy `u = kp (v* − v) + d c + n`, latent AR(1) with persistence ρ
(or the alternative latent processes of Experiment 2). Δ = 0.1, T = 60, b = 1, k = 0.3, g = 1, kp = 0.8,
v* = 20, σ_c = 1, σ_n = 0.5, σ_w = 0.05; `d` and `ρ` vary as stated in the paper. The interventional query is
E[v_T | do(u = ū), v_0 = v*] on ū ∈ [3, 6] (in-support) and ū ∈ [7, 9] (out-of-support).

## License
MIT (code). The paper text and figures are © the author.
