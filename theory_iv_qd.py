#!/usr/bin/env python3
"""
Quasi-differenced instrumental-variable (GMM) estimator — the classical remedy for a lagged dependent variable with
AR(1) errors (Hatanaka 1974; the dynamic-panel GMM literature), applied to the paper's closed loop.

Residual with the dynamics written as   r_t := (v_{t+1} - v_t)/Δ = b u_t - k v_t + g c_t + w_t/Δ.
Quasi-differencing removes the AR(1) latent:   r_t - ρ r_{t-1} = b(u_t - ρ u_{t-1}) - k(v_t - ρ v_{t-1}) + e_t,
       e_t = g sqrt(1-ρ²) ξ_t + (w_t - ρ w_{t-1})/Δ,
and e_t is independent of (v_{t-1}, v_{t-2}, u_{t-1}, u_{t-2}) (all functions of innovations dated t-2 and earlier,
plus n_{t-1}, n_{t-2}, which do not enter e_t).  Moment conditions E[z_t e_t(θ)] = 0, θ = (b, k, ρ) (or (k, ρ) with b known),
z_t = (1, v_{t-1}, v_{t-2}, u_{t-1}, u_{t-2}); solved by two-step GMM.  Compared with the Anderson–Hsiao estimator of
theory_iv_baseline.py, which differences only once and is valid at ρ = 1 (and ρ = 0) but not between.
"""
import json, os, numpy as np
from scipy.optimize import least_squares
import exp1_confounded_vehicle as E

HERE = os.path.dirname(os.path.abspath(__file__)); P = E.P; DT = P['dt']

def moments(theta, v, u, b_known):
    if b_known is None: b, k, rho = theta
    else: b = b_known; k, rho = theta
    r = (v[:, 1:] - v[:, :-1])/DT                                        # r_t, t = 0..T-1
    # need t >= 2 for the instruments
    e = (r[:, 2:] - rho*r[:, 1:-1]) - b*(u[:, 2:] - rho*u[:, 1:-1]) + k*(v[:, 2:-1] - rho*v[:, 1:-2])
    z = np.stack([np.ones_like(e), v[:, 1:-2], v[:, :-3], u[:, 1:-1], u[:, :-2]], -1)
    return (z*e[..., None]).reshape(-1, z.shape[-1]).mean(0), z, e

def gmm(v, u, b_known=None, init=None):
    th0 = np.array([0.5, 0.1, 0.5]) if b_known is None else np.array([0.1, 0.5])
    if init is not None: th0 = np.array(init)
    W = np.eye(5)
    for step in range(2):
        f = lambda th: np.linalg.cholesky(W) @ moments(th, v, u, b_known)[0]
        th = least_squares(f, th0).x
        _, z, e = moments(th, v, u, b_known); g = (z*e[..., None]).reshape(-1, 5)
        W = np.linalg.inv(np.cov(g.T) + 1e-9*np.eye(5)); th0 = th
    return th

if __name__ == "__main__":
    out = []
    print(" ρ    |  b free: (b̂, k̂, ρ̂)        |  b known: (k̂, ρ̂)")
    for rho in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]:
        v, u, c = E.simulate(1.0, rho, n_traj=20000, seed=3)
        th3 = gmm(v, u); th2 = gmm(v, u, b_known=P['b'], init=th3[1:])     # b-known fit started from the free solution
        out.append(dict(rho=rho, free=th3.tolist(), known=th2.tolist()))
        print(f"{rho:4.2f} |  ({th3[0]:.3f}, {th3[1]:.3f}, {th3[2]:.3f})   |  ({th2[0]:.3f}, {th2[1]:.3f})")
    json.dump(out, open(os.path.join(HERE, "theory_iv_qd.json"), "w"), indent=1)
