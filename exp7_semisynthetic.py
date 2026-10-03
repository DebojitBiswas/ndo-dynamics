#!/usr/bin/env python3
"""
Experiment 7 — a semi-synthetic car-following world calibrated on NGSIM US-101, with interventional ground truth.

Everything that can be taken from the data is:  initial states (v0, gap0), the library of real leader speed
profiles, the driver's observational response pî(v, gap, v_L) (an MLP fitted to NGSIM), the total variance of the
driver's unexplained acceleration, and the latent persistence (rho = 0.80, the prior the latent-process model
learned on the real data in Exp. 6).  What the data cannot give — the strength lambda of the confounding link
between the driver's latent state and the leader's behaviour — is the experimental knob.

World (Δ = 0.1 s, T = 60):
    c_t   = rho c_{t-1} + sqrt(1-rho²) xi_t                         latent driver/traffic state
    v_L   = a real NGSIM leader profile, chosen so that corr(rank of braking severity, c_0) = lambda
    a_t   = pî(v_t, gap_t, v_{L,t}) + d c_t + n_t                   driver's acceleration (policy)
    v'    = v + Δ a                                                 exact unit-gain row (P1)
    gap'  = gap + Δ (v_L − v)                                       exact kinematic row (P2)
Query:  E[v_T | do(v_L = profile), v_0, gap_0] for 100 held-out real braking profiles; ground truth by
simulation with c drawn from its prior.  Estimators: NAIVE, DKF (free rows), DKF-P2, DKF-P12 of realdata_ngsim.py.
Usage:  EXP7_SEED=0 python3 exp7_semisynthetic.py /path/to/ngsim_us101_slim.csv
"""
import json, math, os, sys, time, numpy as np, torch
import realdata_ngsim as R
import exp2_latent_process_ndo as X

torch.set_num_threads(int(os.environ.get('EXP2_THREADS', 1)))
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, f"exp7_results_rho{os.environ.get('EXP7_RHO', '0.80')}.json")
DT, T = R.DT, R.T; SEED = int(os.environ.get('EXP2_SEED', 0)); RHO = float(os.environ.get('EXP7_RHO', 0.80)); LAMBDAS = [float(x) for x in os.environ.get('EXP7_LAMBDAS', '0.0,0.6').split(',')]

_PI = []
def fit_policy(ws, seed=0):
    v, gap, vl, a = R.to_tensors(ws); torch.manual_seed(seed); pi = X.mlp(3, 1, 64); opt = torch.optim.Adam(pi.parameters(), 2e-3)
    f = lambda v, g, l: torch.stack([v/10, g/20, (l - v)/5], -1)
    Xf = f(v[:, :-1], gap[:, :-1], vl[:, :-1]); Y = a[:, :-1]
    for it in range(3000):
        idx = torch.randint(0, Xf.shape[0], (256,)); loss = ((pi(Xf[idx])[..., 0]*2 - Y[idx])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): resid_var = float(((Y - pi(Xf)[..., 0]*2)**2).mean())
    _PI.append(pi)
    return (lambda v, g, l: pi(f(v, g, l))[..., 0]*2), resid_var

@torch.no_grad()
def simulate(n, pi_hat, sig2, profiles, inits, lam, rng, c_fixed=None, prof_idx=None):
    """profiles: (M, T+1) real leader paths; inits: (M, 2) matching (v0, gap0). Returns v, gap, vl, a, c."""
    d = math.sqrt(0.5*sig2); sn = math.sqrt(0.5*sig2)                      # half the unexplained variance persistent, half white
    c = np.zeros((n, T+1)); c[:, 0] = rng.normal(size=n) if c_fixed is None else c_fixed
    for t in range(1, T+1): c[:, t] = RHO*c[:, t-1] + math.sqrt(1 - RHO**2)*rng.normal(size=n)
    if prof_idx is None:
        # choose a profile whose braking-severity rank correlates with c_0:  z = lam*c0 + sqrt(1-lam²) eps  →  quantile
        z = lam*c[:, 0] + math.sqrt(1 - lam**2)*rng.normal(size=n); q = 0.5*(1 + torch.erf(torch.tensor(z)/math.sqrt(2)).numpy())
        sev = profiles[:, 0] - profiles.min(1); order = np.argsort(sev)                  # ascending severity
        prof_idx = order[np.clip((q*len(order)).astype(int), 0, len(order) - 1)]
    vl = profiles[prof_idx]; v = np.zeros((n, T+1)); gap = np.zeros((n, T+1)); a = np.zeros((n, T+1))
    v[:, 0] = inits[prof_idx, 0]; gap[:, 0] = inits[prof_idx, 1]
    for t in range(T):
        a[:, t] = pi_hat(torch.tensor(v[:, t], dtype=torch.float32), torch.tensor(gap[:, t], dtype=torch.float32), torch.tensor(vl[:, t], dtype=torch.float32)).numpy() + d*c[:, t] + sn*rng.normal(size=n)
        v[:, t+1] = v[:, t] + DT*a[:, t]; gap[:, t+1] = gap[:, t] + DT*(vl[:, t] - v[:, t])
    a[:, T] = a[:, T-1]
    return v, gap, vl, a, c, prof_idx

def ground_truth(pi_hat, sig2, profiles, inits, q_idx, rng, n_mc=400):
    """E[v_{1:T} | do(v_L = profile_j), v0, gap0] for each query profile j, c from the prior."""
    out = []
    for j in q_idx:
        v, *_ = simulate(n_mc, pi_hat, sig2, profiles, inits, 0.0, rng, prof_idx=np.full(n_mc, j))
        out.append(v[:, 1:].mean(0))
    return np.stack(out)

def tens(*arrs): return [torch.tensor(x, dtype=torch.float32) for x in arrs]

if __name__ == "__main__":
    csv = sys.argv[1]; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ws = R.load_pairs(csv); rng = np.random.default_rng(100 + SEED)
    profiles = np.stack([w['vl'] for w in ws]); inits = np.stack([[w['v'][0], w['gap'][0]] for w in ws])
    brake = np.where((profiles.min(1) - profiles[:, 0]) < -3.0)[0]; q_idx = rng.choice(brake, 100, replace=False)   # query: real braking profiles
    cache = os.path.join(HERE, f"exp7_cache_rho{RHO}_seed{SEED}.pt")
    if os.path.exists(cache):
        ck = torch.load(cache, weights_only=False); pi_net, sig2, gt = ck['pi'], ck['sig2'], ck['gt']
        f = lambda v, g, l: torch.stack([v/10, g/20, (l - v)/5], -1); pi_hat = lambda v, g, l: pi_net(f(v, g, l))[..., 0]*2
    else:
        pi_hat, sig2 = fit_policy(ws, seed=SEED); print(f"fitted observational policy; unexplained acceleration variance {sig2:.3f}")
        gt = ground_truth(pi_hat, sig2, profiles, inits, q_idx, rng)
        torch.save(dict(pi=pi_hat.__closure__[0].cell_contents if False else _PI[0], sig2=sig2, gt=gt), cache)
    print("ground truth: mean |v_T - v_0| =", np.abs(gt[:, -1] - inits[q_idx, 0]).mean().round(2))
    for lam in LAMBDAS:
        v, gap, vl, a, c, _ = simulate(4000, pi_hat, sig2, profiles, inits, lam, rng)
        print(f"lambda={lam}: corr(c0, braking severity) = {np.corrcoef(c[:, 0], vl[:, 0] - vl.min(1))[0, 1]:.2f}")
        tr = tens(v, gap, vl, a)
        for mode in ['naive', 'free', 'p2', 'phys']:
            key = f"{mode}:lam{lam}:seed{SEED}"
            if key in res: continue
            t0 = time.time(); m = R.RealDKF(mode); R.train(m, tr, steps=3000, seed=SEED)
            with torch.no_grad():
                pred = m.rollout(torch.tensor(inits[q_idx, 0], dtype=torch.float32), torch.tensor(inits[q_idx, 1], dtype=torch.float32), torch.tensor(profiles[q_idx, :-1], dtype=torch.float32)).numpy()
            r = dict(rmse_T=float(np.sqrt(np.mean((pred[:, -1] - gt[:, -1])**2))), rmse_path=float(np.sqrt(np.mean((pred - gt)**2))), rho=float(torch.sigmoid(m.rho_raw)))
            res[key] = r; json.dump(res, open(OUT, "w"), indent=1); print(f"[{key}] ({time.time()-t0:.0f}s)", r)
