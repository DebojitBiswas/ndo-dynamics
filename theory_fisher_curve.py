#!/usr/bin/env python3
"""
Per-trajectory Fisher information for b and k as a function of latent persistence ρ (Conjecture clauses 3–4).

I(θ) ≈ (1/N) · Hessian of the negative log-likelihood at the TRUE θ, using the exact Kalman likelihood of
theory_kalman_ml.py on N simulated trajectories (law of large numbers: the observed information at the truth
converges to the Fisher information).  Reported: I_b (b free), the asymptotic standard error of b̂ per
trajectory 1/sqrt([I^{-1}]_bb), and the same for k with and without b fixed, on a dense ρ grid.
"""
import json, os, math, numpy as np, torch
import exp1_confounded_vehicle as E
import theory_kalman_ml as K

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__)); P = E.P
TRUE = dict(b=1.0, k=0.3, g=1.0, kp=0.8, v_star=20.0, d=1.0, rho=None, sig_n=0.5, sig_w=0.05)

def info_at_truth(rho, N=3000, seed=5, b_known=None):
    v, u, c = E.simulate(1.0, rho, n_traj=N, seed=seed)
    v = torch.tensor(v); u = torch.tensor(u)
    th = dict(TRUE); th['rho'] = min(max(rho, 1e-6), 1 - 1e-6)
    raw = K.pack(th, b_known)
    H = torch.autograd.functional.hessian(lambda r: K.negloglik(r, v, u, b_known), raw)/N
    # delta method to constrained parameters
    names = [n for n in K.NAMES if not (b_known is not None and n == 'b')]
    def constrained(r):
        t = K.unpack(r, b_known); return torch.stack([t[n] for n in names])
    Jc = torch.autograd.functional.jacobian(constrained, raw)
    cov = Jc @ torch.linalg.inv(H) @ Jc.T          # asymptotic covariance per trajectory
    I = torch.linalg.inv(cov)                       # Fisher information in constrained coordinates
    out = {n: dict(I=float(I[i, i]), se1=float(torch.sqrt(cov[i, i]))) for i, n in enumerate(names)}
    if b_known is None and 'b' in names:
        # b-known quantities from the SAME information matrix (Schur complement): delete b's row and column of I.
        keep = [i for i, n in enumerate(names) if n != 'b']; Ik = I[keep][:, keep]; covk = torch.linalg.inv(Ik)
        out['_known'] = {n: dict(I=float(Ik[j, j]), se1=float(torch.sqrt(covk[j, j]))) for j, n in enumerate([names[i] for i in keep])}
    return out

if __name__ == "__main__":
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995, 1.0]
    res = []
    print(" ρ     | I_b     se1(b)  | I_k(b free)  se1(k) | I_k(b known)  se1(k)")
    for rho in grid:
        f = info_at_truth(rho); fk = f['_known']          # same Hessian, b removed: guarantees se_k_known <= se_k_free
        res.append(dict(rho=rho, I_b=f['b']['I'], se_b=f['b']['se1'], I_k_free=f['k']['I'], se_k_free=f['k']['se1'], I_k_known=fk['k']['I'], se_k_known=fk['k']['se1']))
        print(f" {rho:5.3f} | {f['b']['I']:8.1f} {f['b']['se1']:7.4f} | {f['k']['I']:10.1f} {f['k']['se1']:7.4f} | {fk['k']['I']:10.1f} {fk['k']['se1']:7.4f}")
    json.dump(res, open(os.path.join(HERE, "theory_fisher_curve.json"), "w"), indent=1)
