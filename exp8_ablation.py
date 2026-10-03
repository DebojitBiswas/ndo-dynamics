#!/usr/bin/env python3
"""
Experiment 8 — head ablation: where does a learned latent-process model lose the input gain?

Hypothesis A (policy head): the free policy head pi_theta(v, c) lacks the parametric structure (set-point term)
through which the linear estimator identifies b at low persistence.
Hypothesis B (dynamics head): a free dynamics head f_theta(v, u, c), having a free intercept and a free function,
gives up the mean relation that pins b down; the policy head is irrelevant.

Variants (AR(1) latent, d = 1; worlds rho = 0.3 and rho = 0.9; seeds 11, 21):
    PPOL-FDYN : parametric policy  u = kp (v* - v) + d c  (kp, v*, d learned);  free dynamics MLP f(v, u, c)
    FPOL-FDYN : = DKF of Exp. 2/3 (both free)                       [results already in exp3/exp2 json]
    FPOL-PDYN : = DKF-P2 of Exp. 4 (free policy, parametric dynamics, b learned)   [exp4 json]
    PPOL-PDYN : both parametric (the neural Kalman-ML)
Reported: learned gain d f/d u at data points (true 1), do-RMSE in / out of support.
Usage: EXP2_SEED=11 python3 exp8_ablation.py
"""
import json, math, os, sys, time, numpy as np, torch, torch.nn as nn
import exp1_confounded_vehicle as E
import exp2_latent_process_ndo as X
import exp4_p2_and_sensitivity as X4

torch.set_num_threads(int(os.environ.get('EXP2_THREADS', 1)))
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "exp8_results.json")
P = E.P; DT = P['dt']; T = P['T']; SEED = int(os.environ.get('EXP2_SEED', 11))

class DKFAbl(X.DKF):
    def __init__(self, pol='P', dyn='F', h=64):
        super().__init__(known_b=None, h=h); self.pol, self.dynk = pol, dyn
        if pol == 'P':
            self.kp_raw = nn.Parameter(torch.tensor(0.5)); self.vs_raw = nn.Parameter(torch.tensor(18.0)); self.d_raw = nn.Parameter(torch.tensor(0.5))
        if dyn == 'P':
            self.b_raw = nn.Parameter(torch.tensor(0.5)); self.k_raw = nn.Parameter(torch.tensor(0.1)); self.g_raw = nn.Parameter(torch.tensor(0.5))
    def policy(self, v, c):
        if self.pol == 'P': return self.kp_raw*(self.vs_raw - v) + self.d_raw*c
        return super().policy(v, c)
    def dyn(self, v, u, c):
        if self.dynk == 'P': return v + DT*(self.b_raw*u - self.k_raw*v + self.g_raw*c)
        return super().dyn(v, u, c)

if __name__ == "__main__":
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for world in ['RHO0.3', 'AR1']:
        v, u, c, gt, gt_ood = X4.make_world(world)
        for name, (pol, dyn) in [('PPOL-FDYN', ('P', 'F')), ('PPOL-PDYN', ('P', 'P'))]:
            key = f"{name}:{world}:seed{SEED}"
            if key in res: continue
            t0 = time.time(); torch.manual_seed(SEED); m = DKFAbl(pol, dyn); L = X.train_dkf(m, v, u, seed=SEED)
            r, ro, cur = X4.evaluate(m, gt, gt_ood)
            with torch.no_grad():
                x = torch.stack([X.nv(torch.tensor(v[:, :-1], dtype=torch.float32)), X.nu(torch.tensor(u, dtype=torch.float32)), X.nv(torch.tensor(v[:, 1:], dtype=torch.float32))], -1)
                hdn, _ = m.enc(x); mu = m.enc_out(hdn)[..., 0]
            out = dict(rmse=r, rmse_ood=ro, dfdu=X.dfdu(m, v, u, mu), rho=float(torch.sigmoid(m.rho_raw)))
            if pol == 'P': out['policy'] = dict(kp=float(m.kp_raw), v_star=float(m.vs_raw), d=float(m.d_raw))
            if dyn == 'P': out['dyn'] = dict(b=float(m.b_raw), k=float(m.k_raw), g=float(m.g_raw))
            res[key] = out; json.dump(res, open(OUT, "w"), indent=1)
            print(f"[{key}] ({time.time()-t0:.0f}s) rmse {r:.2f}/{ro:.2f} dfdu {out['dfdu']:.3f} rho {out['rho']:.2f}", out.get('policy', ''), out.get('dyn', ''))
