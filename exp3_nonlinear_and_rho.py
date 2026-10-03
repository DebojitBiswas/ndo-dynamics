#!/usr/bin/env python3
"""
Experiment 3 — two additions requested by the paper plan.

(A) Nonlinear dynamics.  Same driver and AR(1) latent (ρ = 0.9, d = 1) but a nonlinear drag:
        v' = v + Δ( b u − k v − k2 v|v|/v* + g c ) + w ,   k2 = 0.15
    Ground truth E[v_T | do(u=ū)] is computed by Monte Carlo over the latent (no closed form now).
    Estimators: NAIVE, NDO-Z, DKF, DKF-B (all neural).  Tests whether the picture survives nonlinear physics.

(B) Persistence sweep for the latent-process models (AR(1) latent, d = 1): DKF and DKF-B at
    ρ ∈ {0.99, 0.7, 0.3}, to be read together with ρ = 0.9 from exp2.  Mirrors panel (c) of exp1.

Usage:  python3 exp3_nonlinear_and_rho.py NL         (part A)
        python3 exp3_nonlinear_and_rho.py RHO 0.99   (part B, one ρ)
Results appended to exp3_results.json (keys 'NL' or 'RHO:<ρ>').
"""
import json, math, os, sys, time, numpy as np, torch
import exp1_confounded_vehicle as E
import exp2_latent_process_ndo as X

HERE = os.path.dirname(os.path.abspath(__file__))
P = E.P; DT = P['dt']; T = P['T']; K2 = 0.15
OUT = os.path.join(HERE, "exp3_results.json")
SEED = int(os.environ.get('EXP2_SEED', 11))

def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(d): json.dump(d, open(OUT, "w"), indent=1)

# ---------------------------------------------------------------- (A) nonlinear world
def drag(v): return P['k']*v + K2*v*np.abs(v)/P['v_star']

def simulate_nl(n, seed, rho=0.9, d=1.0):
    rng = np.random.default_rng(seed); c = X.latent_paths('AR1', n, rng) if rho == 0.9 else None
    v = np.zeros((n, T+1)); u = np.zeros((n, T)); v[:, 0] = P['v_star'] + rng.normal(0, 1, n)
    for t in range(T):
        u[:, t] = P['kp']*(P['v_star'] - v[:, t]) + d*c[:, t] + rng.normal(0, P['sig_n'], n)
        v[:, t+1] = v[:, t] + DT*(P['b']*u[:, t] - drag(v[:, t]) + P['g']*c[:, t]) + rng.normal(0, P['sig_w'], n)
    return v, u, c

def gt_nl(u_bar, n=20000, seed=99):
    rng = np.random.default_rng(seed); c = X.latent_paths('AR1', n, rng)
    v = np.full(n, float(P['v_star']))
    for t in range(T):
        v = v + DT*(P['b']*u_bar - drag(v) + P['g']*c[:, t]) + rng.normal(0, P['sig_w'], n)
    return float(v.mean())

def run_nl():
    res = load(); key = f"NL:seed{SEED}"
    if key in res: print("done already"); return
    grid = X.U_GRID; GT = np.array([gt_nl(ub) for ub in grid])
    print("NL ground truth:", np.round(GT, 2))
    v, u, c = simulate_nl(1500, SEED); torch.manual_seed(SEED); out = {'gt': GT.tolist()}
    def rm(fn): cur = np.array([fn(ub) for ub in grid]); return float(np.sqrt(np.mean((cur-GT)**2))), cur.tolist()
    m = X.fit_onestep(v, u, 0); out['NAIVE'] = dict(zip(['rmse', 'curve'], rm(lambda ub: E.rollout_do(m, ub))))
    m = X.fit_onestep(v, u, 1); out['NDO-Z'] = dict(zip(['rmse', 'curve'], rm(lambda ub: E.rollout_do(m, ub))))
    for name, kb in [('DKF', None), ('DKF-B', P['b'])]:
        model = X.DKF(known_b=kb); L = X.train_dkf(model, v, u, seed=SEED)
        r, cur = rm(model.do_query)
        with torch.no_grad():
            xx = torch.stack([X.nv(torch.tensor(v[:, :-1], dtype=torch.float32)), X.nu(torch.tensor(u, dtype=torch.float32)), X.nv(torch.tensor(v[:, 1:], dtype=torch.float32))], -1)
            hdn, _ = model.enc(xx); mu = model.enc_out(hdn)[..., 0]
        out[name] = dict(rmse=r, curve=cur, rho=float(torch.sigmoid(model.rho_raw)), corr_c=float(np.corrcoef(mu.numpy().ravel(), c.ravel())[0, 1]), dfdu=X.dfdu(model, v, u, mu))
    res = load(); res[key] = out; save(res)
    print(f"[NL seed {SEED}] " + "  ".join(f"{k}: {out[k]['rmse']:.2f}" for k in ['NAIVE', 'NDO-Z', 'DKF', 'DKF-B']), " dfdu:", round(out['DKF']['dfdu'], 3))

# ---------------------------------------------------------------- (B) rho sweep for DKF / DKF-B
def run_rho(rho):
    res = load(); key = f"RHO:{rho}:seed{SEED}"
    if key in res: print("done already"); return
    rng = np.random.default_rng(SEED); n = 1500
    c = np.zeros((n, T)); c[:, 0] = rng.normal(0, 1, n)
    for t in range(1, T): c[:, t] = rho*c[:, t-1] + math.sqrt(1-rho**2)*rng.normal(0, 1, n)
    v = np.zeros((n, T+1)); u = np.zeros((n, T)); v[:, 0] = P['v_star'] + rng.normal(0, 1, n)
    for t in range(T):
        u[:, t] = P['kp']*(P['v_star'] - v[:, t]) + 1.0*c[:, t] + rng.normal(0, P['sig_n'], n)
        v[:, t+1] = v[:, t] + DT*(P['b']*u[:, t] - P['k']*v[:, t] + P['g']*c[:, t]) + rng.normal(0, P['sig_w'], n)
    torch.manual_seed(SEED); out = {}
    for name, kb in [('DKF', None), ('DKF-B', P['b'])]:
        model = X.DKF(known_b=kb); L = X.train_dkf(model, v, u, seed=SEED)
        r, cur = X.curve_rmse(model.do_query); r_ood = X.ood_rmse(model.do_query)
        with torch.no_grad():
            xx = torch.stack([X.nv(torch.tensor(v[:, :-1], dtype=torch.float32)), X.nu(torch.tensor(u, dtype=torch.float32)), X.nv(torch.tensor(v[:, 1:], dtype=torch.float32))], -1)
            hdn, _ = model.enc(xx); mu = model.enc_out(hdn)[..., 0]
        out[name] = dict(rmse=r, rmse_ood=r_ood, curve=cur, rho=float(torch.sigmoid(model.rho_raw)), dfdu=X.dfdu(model, v, u, mu))
    res = load(); res[key] = out; save(res)
    print(f"[RHO {rho} seed {SEED}]  DKF: {out['DKF']['rmse']:.2f} (ood {out['DKF']['rmse_ood']:.2f}, dfdu {out['DKF']['dfdu']:.2f})   DKF-B: {out['DKF-B']['rmse']:.2f} (ood {out['DKF-B']['rmse_ood']:.2f})")

if __name__ == "__main__":
    if sys.argv[1] == 'NL': run_nl()
    elif sys.argv[1] == 'RHO': run_rho(float(sys.argv[2]))
