#!/usr/bin/env python3
"""
Experiment 4 — closing the two remaining objections.

(A) P2: a physics-structured ("grey-box") state channel.  The latent-process NDO keeps its learned
    policy head, encoder and AR(1) prior, but the dynamics are the known parametric form with unknown
    coefficients:
        f(v,u,c) = b u - k v - k2 v|v|/v* + g c          (b, k, k2, g learned; or b fixed = P1+P2)
    This constrains extrapolation in v, which P1 alone does not.  Variants: DKF-P2 (b learned), DKF-P12 (b fixed).
    Worlds: AR1 (ρ=0.9, linear; true k2 = 0), NL (nonlinear drag), RHO0.3 (low persistence).  In-support + OOD.

(B) Sensitivity of P1 to a wrong assumed gain: DKF-B with b_assumed ∈ {0.8, 0.9, 1.1, 1.2} (true 1.0), AR1 ρ=0.9.

Usage: python3 exp4_p2_and_sensitivity.py P2 <AR1|NL|RHO0.3>
       python3 exp4_p2_and_sensitivity.py WRONGB <b_assumed>
Results appended to exp4_results.json.
"""
import json, math, os, sys, time, numpy as np, torch, torch.nn as nn
import exp1_confounded_vehicle as E
import exp2_latent_process_ndo as X
import exp3_nonlinear_and_rho as X3

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "exp4_results.json")
P = E.P; DT = P['dt']; T = P['T']; SEED = int(os.environ.get('EXP2_SEED', 11))
def load(): return json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(d): json.dump(d, open(OUT, "w"), indent=1)

class DKFP2(X.DKF):
    """Grey-box dynamics: known functional form, unknown coefficients."""
    def __init__(self, known_b=None, h=64):
        super().__init__(known_b=known_b, h=h)
        self.f = None
        self.b_raw = nn.Parameter(torch.tensor(0.5)); self.k_raw = nn.Parameter(torch.tensor(0.1))
        self.k2_raw = nn.Parameter(torch.tensor(0.0)); self.g_raw = nn.Parameter(torch.tensor(0.5))
    def dyn(self, v, u, c):
        b = self.b_raw if self.known_b is None else torch.as_tensor(float(self.known_b))
        return v + DT*(b*u - self.k_raw*v - self.k2_raw*v*torch.abs(v)/P['v_star'] + self.g_raw*c)
    def coeffs(self):
        return dict(b=float(self.b_raw) if self.known_b is None else float(self.known_b), k=float(self.k_raw), k2=float(self.k2_raw), g=float(self.g_raw))

def make_world(name):
    if name == 'AR1':
        v, u, c = X.simulate('AR1', 1500, seed=SEED); gt = X.GT; gt_ood = X.GT_OOD
    elif name == 'NL':
        v, u, c = X3.simulate_nl(1500, SEED); gt = np.array([X3.gt_nl(ub) for ub in X.U_GRID]); gt_ood = np.array([X3.gt_nl(ub) for ub in X.U_OOD])
    elif name.startswith('RHO'):
        rho = float(name[3:]); rng = np.random.default_rng(SEED); n = 1500
        c = np.zeros((n, T)); c[:, 0] = rng.normal(0, 1, n)
        for t in range(1, T): c[:, t] = rho*c[:, t-1] + math.sqrt(1-rho**2)*rng.normal(0, 1, n)
        v = np.zeros((n, T+1)); u = np.zeros((n, T)); v[:, 0] = P['v_star'] + rng.normal(0, 1, n)
        for t in range(T):
            u[:, t] = P['kp']*(P['v_star'] - v[:, t]) + c[:, t] + rng.normal(0, P['sig_n'], n)
            v[:, t+1] = v[:, t] + DT*(P['b']*u[:, t] - P['k']*v[:, t] + P['g']*c[:, t]) + rng.normal(0, P['sig_w'], n)
        gt = X.GT; gt_ood = X.GT_OOD
    return v, u, c, gt, gt_ood

def evaluate(model, gt, gt_ood):
    cur = np.array([model.do_query(ub) for ub in X.U_GRID]); cur_o = np.array([model.do_query(ub) for ub in X.U_OOD])
    return float(np.sqrt(np.mean((cur-gt)**2))), float(np.sqrt(np.mean((cur_o-gt_ood)**2))), cur.tolist()

def run_p2(world):
    res = load(); v, u, c, gt, gt_ood = make_world(world)
    for name, kb in [('DKF-P2', None), ('DKF-P12', P['b'])]:
        key = f"P2:{world}:{name}:seed{SEED}"
        if key in res: continue
        torch.manual_seed(SEED); m = DKFP2(known_b=kb); L = X.train_dkf(m, v, u, seed=SEED)
        r, ro, cur = evaluate(m, gt, gt_ood)
        res = load(); res[key] = dict(rmse=r, rmse_ood=ro, curve=cur, coeffs=m.coeffs(), rho=float(torch.sigmoid(m.rho_raw))); save(res)
        print(f"[{key}] in {r:.2f}  ood {ro:.2f}  coeffs {m.coeffs()}  rho_hat {float(torch.sigmoid(m.rho_raw)):.2f}")

def run_wrongb(b_assumed):
    res = load(); key = f"WRONGB:{b_assumed}:seed{SEED}"
    if key in res: return
    v, u, c, gt, gt_ood = make_world('AR1'); torch.manual_seed(SEED)
    m = X.DKF(known_b=b_assumed); L = X.train_dkf(m, v, u, seed=SEED); r, ro, cur = evaluate(m, gt, gt_ood)
    res = load(); res[key] = dict(rmse=r, rmse_ood=ro, curve=cur); save(res)
    print(f"[{key}] in {r:.2f}  ood {ro:.2f}")

if __name__ == "__main__":
    if sys.argv[1] == 'P2': run_p2(sys.argv[2])
    elif sys.argv[1] == 'WRONGB': run_wrongb(float(sys.argv[2]))
