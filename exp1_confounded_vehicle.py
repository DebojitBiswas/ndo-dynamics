#!/usr/bin/env python3
"""
Experiment 1 — Interventional identifiability in a confounded dynamical system.

System (discrete time, dt):
    v_{t+1} = v_t + dt * ( b*u_t - k*v_t + g*c ) + w_t          (longitudinal vehicle dynamics)
    u_t     = kp*(v_star - v_t) + d*c + n_t                      (driver policy, closed loop)
    c       ~ N(0, sig_c^2) per trajectory (latent: grade / load / driver trait)
    OR c_t  = rho*c_{t-1} + sqrt(1-rho^2)*xi_t  (time-varying latent process)

The driver *sees* c (anticipates the hill) so u is correlated with c even after
conditioning on v_t.  A one-step neural simulator trained on (v_t, u_t) -> v_{t+1}
learns  E[v_{t+1} | v_t, u_t]  whose coefficient on u is

        b_obs = b + g * d*sig_c^2 / (d^2*sig_c^2 + sig_n^2)       (omitted-variable bias)

so its rollouts under do(u = u_bar) are wrong, and the error compounds over T.

Estimators compared on the do-query  E[v_T | do(u_t = u_bar for all t), v_0 = v_star]:
    GT    : analytic ground truth (linear recursion, c integrated out, E[c]=0)
    NAIVE : MLP one-step simulator, rolled out with u clamped
    NDO-Z : MLP one-step simulator with a learned per-trajectory latent z_i
            (abduction of the trajectory-level confounder); do-query marginalises
            z over the learned population, u clamped (gate)
    NDO-ZB: same as NDO-Z but with the input channel gain b known from physics
            (v' = v + dt*b*u + dt*h_theta(v, z))

Panel A: bias vs confounding strength d, constant latent (rho = 1).
Panel B: bias vs latent persistence rho at fixed d — where trajectory-level
         abduction stops working and physics knowledge has to take over.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(max(1, os.cpu_count() // 2))
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- world
P = dict(dt=0.1, T=60, b=1.0, k=0.3, g=1.0, kp=0.8, v_star=20.0,
         sig_c=1.0, sig_n=0.5, sig_w=0.05, n_traj=1500)

def simulate(d, rho=1.0, n_traj=None, seed=0, u_override=None, c_override=None):
    """Return v [n,T+1], u [n,T], c [n,T] (c constant if rho==1)."""
    rng = np.random.default_rng(seed)
    n = n_traj or P['n_traj']; T, dt = P['T'], P['dt']
    v = np.zeros((n, T+1)); u = np.zeros((n, T)); c = np.zeros((n, T))
    v[:, 0] = P['v_star'] + rng.normal(0, 1.0, n)
    if c_override is not None:
        c[:] = c_override[:, None]
    else:
        c0 = rng.normal(0, P['sig_c'], n)
        c[:, 0] = c0
        for t in range(1, T):
            c[:, t] = rho*c[:, t-1] + math.sqrt(max(0.0, 1-rho**2))*rng.normal(0, P['sig_c'], n)
    for t in range(T):
        if u_override is None:
            u[:, t] = P['kp']*(P['v_star'] - v[:, t]) + d*c[:, t] + rng.normal(0, P['sig_n'], n)
        else:
            u[:, t] = u_override
        v[:, t+1] = v[:, t] + dt*(P['b']*u[:, t] - P['k']*v[:, t] + P['g']*c[:, t]) \
                    + rng.normal(0, P['sig_w'], n)
    return v, u, c

def gt_do(u_bar):
    """Analytic E[v_T | do(u=u_bar), v_0=v*]; E[c_t]=0 so c drops out of the mean."""
    v = P['v_star']
    for _ in range(P['T']):
        v = v + P['dt']*(P['b']*u_bar - P['k']*v)
    return v

def analytic_bias_coeff(d):
    sc2, sn2 = P['sig_c']**2, P['sig_n']**2
    return P['g']*d*sc2/(d*d*sc2 + sn2)

# ----------------------------------------------------------------------------- models
def mlp(din, dout, h=64):
    return nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, dout))

class OneStep(nn.Module):
    """v' = v + dt * f(v, u, z).   z: per-trajectory latent (dim zdim, 0 = naive).
       known_b: if not None, the u-channel is fixed to physics: v' = v + dt*(b*u + h(v,z))."""
    def __init__(self, n_traj, zdim=0, known_b=None):
        super().__init__()
        self.zdim, self.known_b = zdim, known_b
        self.z = nn.Embedding(n_traj, zdim) if zdim > 0 else None
        if self.z is not None: nn.init.normal_(self.z.weight, 0, 0.1)
        din = (1 if known_b is not None else 2) + zdim
        self.f = mlp(din, 1)
    def step(self, v, u, z):
        v_s = (v - P['v_star'])/5.0; u_s = u/2.0
        if self.known_b is None:
            inp = [v_s[..., None], u_s[..., None]]
        else:
            inp = [v_s[..., None]]
        if z is not None: inp.append(z)
        f = self.f(torch.cat(inp, -1))[..., 0]
        if self.known_b is not None:
            f = f + self.known_b*u
        return v + P['dt']*f

def train(model, v, u, epochs=1000, lr=3e-3, lam_z=0.0, log=""):
    v = torch.tensor(v, dtype=torch.float32); u = torch.tensor(u, dtype=torch.float32)
    n, T = u.shape
    idx = torch.arange(n)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        z = model.z(idx)[:, None, :].expand(n, T, model.zdim) if model.z is not None else None
        pred = model.step(v[:, :-1], u, z)
        loss = ((pred - v[:, 1:])**2).mean()/P['sig_w']**2
        if model.z is not None: loss = loss + lam_z*(model.z.weight**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return loss.item()

@torch.no_grad()
def rollout_do(model, u_bar, n=4000, seed=123):
    """Interventional rollout: clamp u (gate), marginalise z over the learned population."""
    g = torch.Generator().manual_seed(seed)
    v = torch.full((n,), float(P['v_star']))
    z = None
    if model.z is not None:
        zi = torch.randint(0, model.z.num_embeddings, (n,), generator=g)
        z = model.z(zi)
    u = torch.full((n,), float(u_bar))
    for _ in range(P['T']):
        v = model.step(v, u, z)
    return v.mean().item()

# ----------------------------------------------------------------------------- experiment
U_GRID = np.linspace(3.0, 6.0, 7)
GT = np.array([gt_do(ub) for ub in U_GRID])

def run_setting(d, rho, seed=0, epochs=1000):
    v, u, c = simulate(d, rho, seed=seed)
    n = v.shape[0]
    res = {}
    for name, kw in [("NAIVE", dict(zdim=0)), ("NDO-Z", dict(zdim=1)), ("NDO-ZB", dict(zdim=1, known_b=P['b']))]:
        torch.manual_seed(seed)
        m = OneStep(n, **kw)
        train(m, v, u, epochs=epochs)
        curve = np.array([rollout_do(m, ub) for ub in U_GRID])
        res[name] = dict(curve=curve.tolist(), rmse=float(np.sqrt(np.mean((curve-GT)**2))))
    # analytic one-step OLS bias for reference (linear regression of v' on v,u)
    X = np.stack([v[:, :-1].ravel(), u.ravel(), np.ones(u.size)], 1); y = v[:, 1:].ravel()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res["ols_b_hat"] = float((beta[1])/P['dt'])
    # population omitted-variable bias: regress the omitted term g*c_t on the same regressors
    gam = np.linalg.lstsq(X, P['g']*c.ravel(), rcond=None)[0]
    res["ols_b_theory"] = float(P['b'] + gam[1]); res["ols_k_theory"] = float(-P['k'] + gam[0]); res["ols_k_hat"] = float((beta[0]-1)/P['dt'])
    return res

if __name__ == "__main__":
    t0 = time.time()
    out = dict(params=P, u_grid=U_GRID.tolist(), gt=GT.tolist(), panelA=[], panelB=[])
    print("Panel A: constant latent, sweep confounding strength d")
    for d in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        r = run_setting(d, 1.0); r['d'] = d; out['panelA'].append(r)
        print(f"  d={d:4.2f}  OLS b_hat={r['ols_b_hat']:.3f} (OVB theory {r['ols_b_theory']:.3f})  "
              f"RMSE naive={r['NAIVE']['rmse']:.3f}  ndo-z={r['NDO-Z']['rmse']:.3f}  ndo-zb={r['NDO-ZB']['rmse']:.3f}")
    print("Panel B: d=1.0, sweep latent persistence rho")
    for rho in [1.0, 0.99, 0.95, 0.9, 0.7, 0.3, 0.0]:
        r = run_setting(1.0, rho); r['rho'] = rho; out['panelB'].append(r)
        print(f"  rho={rho:4.2f}  OLS b_hat={r['ols_b_hat']:.3f} (OVB theory {r['ols_b_theory']:.3f})  "
              f"RMSE naive={r['NAIVE']['rmse']:.3f}  ndo-z={r['NDO-Z']['rmse']:.3f}  ndo-zb={r['NDO-ZB']['rmse']:.3f}")
    json.dump(out, open(os.path.join(OUT, "exp1_results.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")
