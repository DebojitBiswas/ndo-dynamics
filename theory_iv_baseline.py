#!/usr/bin/env python3
"""Anderson–Hsiao / Arellano–Bond style IV baseline (known b, first differences, lagged-level instruments).
   Δr_t = -k Δv_t + g Δc_t + noise,  instruments v_{t-2}, v_{t-3}  (2SLS).  Valid iff Cov(v_{t-s}, Δc_t)=0, i.e. ρ=1."""
import json, numpy as np, exp1_confounded_vehicle as E
P = E.P; dt = P['dt']; out = []
for rho in [1.0, 0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.0]:
    v, u, c = E.simulate(1.0, rho, n_traj=20000, seed=3)
    r = (v[:, 1:] - v[:, :-1])/dt - P['b']*u                     # r_t, t=0..T-1
    dr = r[:, 3:] - r[:, 2:-1]; dv = v[:, 3:-1] - v[:, 2:-2]        # Δr_t, Δv_t for t=3..T-1
    Z = np.stack([v[:, 1:-3].ravel(), v[:, :-4].ravel()], 1)        # v_{t-2}, v_{t-3}
    X = dv.ravel()[:, None]; y = dr.ravel()
    Zc = Z - Z.mean(0); Xc = X - X.mean(0); yc = y - y.mean()
    Pz = Zc @ np.linalg.solve(Zc.T @ Zc, Zc.T @ Xc)                  # first stage fitted values
    k_hat = -float((Pz[:, 0] @ yc)/(Pz[:, 0] @ Xc[:, 0]))
    out.append(dict(rho=rho, k_hat=k_hat)); print(f"rho={rho:4.2f}  AH-IV k_hat={k_hat:+.3f}  (true 0.300)")
json.dump(out, open('theory_iv_baseline.json', 'w'), indent=1)
