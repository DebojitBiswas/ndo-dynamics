#!/usr/bin/env python3
"""
Experiment 2 — the missing row of the estimator hierarchy: a *neural latent-process* NDO.

Question: when the latent is NOT the AR(1) the model assumes (misspecified latent dynamics), does a
learned latent-process model still recover the do-query, and does fixing the input gain b (physics P1)
matter for it?

World: same vehicle dynamics and driver policy as exp1 (d = 1), but the latent c_t is either
    AR1   : ρ = 0.9 Gaussian AR(1)                         (well specified for the model's prior)
    PWC   : piecewise-constant 'road segments' — segment lengths ~ Geometric(mean 12 steps),
            segment values ~ N(0,1)                        (non-Gaussian, non-AR(1); same E c = 0)
    SINE  : c_t = A sin(ω t + φ), A ~ N(0,1), ω ~ U(0.05,0.3), φ ~ U(0,2π)   (smooth, deterministic given 3 draws)

Estimators (all neural, all trained on the same observational trajectories):
    NAIVE  : MLP one-step (v,u) -> v'                                           [exp1 row 1]
    NDO-Z  : MLP one-step with per-trajectory latent embedding                   [exp1 row 2]
    DKF    : deep-Kalman NDO.  q(c_{0:T}|v,u) = bidirectional GRU encoder; generative model
             u_t ~ N(π(v_t,c_t), σ_n²),  v_{t+1} ~ N(v_t + Δ f(v_t,u_t,c_t), σ_w²),
             prior c_0 ~ N(0,1), c_{t+1} ~ N(ρ c_t, s²) with learned (ρ, s).  Trained by ELBO.
    DKF-B  : same, but f(v,u,c) = b u + h(v,c) with b fixed at the true value   [physics P1]
Do-query: clamp u = ū (gate), sample c paths from the learned prior, roll out, mean v_T.  Truth from exp1.gt_do.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
import exp1_confounded_vehicle as E

torch.set_num_threads(int(os.environ.get('EXP2_THREADS', max(1, os.cpu_count()//2))))
SEED = int(os.environ.get('EXP2_SEED', 11))
HERE = os.path.dirname(os.path.abspath(__file__))
P = E.P; DT = P['dt']; T = P['T']
U_GRID = np.linspace(3.0, 6.0, 7); GT = np.array([E.gt_do(ub) for ub in U_GRID])
U_OOD = np.linspace(7.0, 9.0, 5); GT_OOD = np.array([E.gt_do(ub) for ub in U_OOD])   # outside the training support (v_T 23-30 m/s)

# ------------------------------------------------------------------ worlds
def latent_paths(kind, n, rng):
    c = np.zeros((n, T))
    if kind == 'AR1':
        rho = 0.9; c[:, 0] = rng.normal(0, 1, n)
        for t in range(1, T): c[:, t] = rho*c[:, t-1] + math.sqrt(1-rho**2)*rng.normal(0, 1, n)
    elif kind == 'PWC':
        for i in range(n):
            t = 0
            while t < T:
                L = rng.geometric(1/12.0); c[i, t:t+L] = rng.normal(0, 1); t += L
    elif kind == 'SINE':
        A = rng.normal(0, 1, n); w = rng.uniform(0.05, 0.3, n); ph = rng.uniform(0, 2*np.pi, n)
        tt = np.arange(T)[None, :]; c = A[:, None]*np.sin(w[:, None]*tt + ph[:, None])
    return c

def simulate(kind, n, seed, d=1.0):
    rng = np.random.default_rng(seed); c = latent_paths(kind, n, rng)
    v = np.zeros((n, T+1)); u = np.zeros((n, T)); v[:, 0] = P['v_star'] + rng.normal(0, 1, n)
    for t in range(T):
        u[:, t] = P['kp']*(P['v_star'] - v[:, t]) + d*c[:, t] + rng.normal(0, P['sig_n'], n)
        v[:, t+1] = v[:, t] + DT*(P['b']*u[:, t] - P['k']*v[:, t] + P['g']*c[:, t]) + rng.normal(0, P['sig_w'], n)
    return v, u, c

# ------------------------------------------------------------------ models
def mlp(din, dout, h=64):
    return nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, dout))

VS, US = 5.0, 2.0   # scales
def nv(v): return (v - P['v_star'])/VS
def nu(u): return u/US

class DKF(nn.Module):
    def __init__(self, known_b=None, h=64):
        super().__init__()
        self.known_b = known_b
        self.enc = nn.GRU(3, h, batch_first=True, bidirectional=True)
        self.enc_out = nn.Linear(2*h, 2)                 # mean, logstd of c_t
        self.pi = mlp(2, 1, h)                           # policy head  (v, c) -> u mean
        self.f = mlp(2 if known_b is not None else 3, 1, h)   # dynamics residual
        self.log_sn = nn.Parameter(torch.tensor(0.0)); self.log_sw = nn.Parameter(torch.tensor(-2.0))
        self.rho_raw = nn.Parameter(torch.tensor(1.0)); self.log_s = nn.Parameter(torch.tensor(-1.0))
    def dyn(self, v, u, c):
        if self.known_b is None:
            return v + DT*self.f(torch.stack([nv(v), nu(u), c], -1))[..., 0]*US
        return v + DT*(self.known_b*u + self.f(torch.stack([nv(v), c], -1))[..., 0]*US)
    def policy(self, v, c): return self.pi(torch.stack([nv(v), c], -1))[..., 0]*US
    def elbo(self, v, u):
        x = torch.stack([nv(v[:, :-1]), nu(u), nv(v[:, 1:])], -1)
        hdn, _ = self.enc(x); mu, ls = self.enc_out(hdn).unbind(-1); ls = ls.clamp(-6, 2)
        c = mu + torch.exp(ls)*torch.randn_like(mu)
        sn, sw = torch.exp(self.log_sn), torch.exp(self.log_sw)
        lp_u = -0.5*(((u - self.policy(v[:, :-1], c))/sn)**2 + 2*torch.log(sn) + math.log(2*math.pi))
        lp_v = -0.5*(((v[:, 1:] - self.dyn(v[:, :-1], u, c))/sw)**2 + 2*torch.log(sw) + math.log(2*math.pi))
        rho, s = torch.sigmoid(self.rho_raw), torch.exp(self.log_s)
        lp_c0 = -0.5*(c[:, 0]**2 + math.log(2*math.pi))
        lp_ct = -0.5*(((c[:, 1:] - rho*c[:, :-1])/s)**2 + 2*torch.log(s) + math.log(2*math.pi))
        lq = -0.5*(((c - mu)/torch.exp(ls))**2 + 2*ls + math.log(2*math.pi))
        return (lp_u.sum(1) + lp_v.sum(1) + lp_c0 + lp_ct.sum(1) - lq.sum(1)).mean()
    @torch.no_grad()
    def do_query(self, u_bar, n=4000, seed=1):
        g = torch.Generator().manual_seed(seed)
        rho, s = torch.sigmoid(self.rho_raw), torch.exp(self.log_s)
        c = torch.randn(n, generator=g); v = torch.full((n,), float(P['v_star'])); u = torch.full((n,), float(u_bar))
        for t in range(T):
            v = self.dyn(v, u, c); c = rho*c + s*torch.randn(n, generator=g)
        return v.mean().item()

def train_dkf(model, v, u, steps=4000, bs=256, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    v = torch.tensor(v, dtype=torch.float32); u = torch.tensor(u, dtype=torch.float32); n = v.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        idx = torch.randint(0, n, (bs,)); loss = -model.elbo(v[idx], u[idx])/T
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
    return loss.item()

def fit_onestep(v, u, zdim, epochs=1000):
    m = E.OneStep(v.shape[0], zdim=zdim); E.train(m, v, u, epochs=epochs); return m

def curve_rmse(fn):
    cur = np.array([fn(ub) for ub in U_GRID]); return float(np.sqrt(np.mean((cur - GT)**2))), cur.tolist()
def ood_rmse(fn):
    cur = np.array([fn(ub) for ub in U_OOD]); return float(np.sqrt(np.mean((cur - GT_OOD)**2)))

@torch.no_grad()
def dfdu(model, v, u, c_hat):
    """mean learned partial derivative of the dynamics wrt u at data points (true b = 1)."""
    vt = torch.tensor(v[:, :-1], dtype=torch.float32); ut = torch.tensor(u, dtype=torch.float32, requires_grad=True)
    with torch.enable_grad():
        out = model.dyn(vt, ut, c_hat); g = torch.autograd.grad(out.sum(), ut)[0]/DT
    return float(g.mean())

if __name__ == "__main__":
    kinds = sys.argv[1:] or ['AR1', 'PWC', 'SINE']
    N = 1500; out = {}
    for kind in kinds:
        v, u, c = simulate(kind, N, seed=SEED); res = {}; torch.manual_seed(SEED)
        t0 = time.time()
        m = fit_onestep(v, u, 0);  res['NAIVE'] = dict(zip(['rmse', 'curve'], curve_rmse(lambda ub: E.rollout_do(m, ub)))); res['NAIVE']['rmse_ood'] = ood_rmse(lambda ub: E.rollout_do(m, ub))
        m = fit_onestep(v, u, 1);  res['NDO-Z'] = dict(zip(['rmse', 'curve'], curve_rmse(lambda ub: E.rollout_do(m, ub)))); res['NDO-Z']['rmse_ood'] = ood_rmse(lambda ub: E.rollout_do(m, ub))
        for name, kb in [('DKF', None), ('DKF-B', P['b'])]:
            model = DKF(known_b=kb); L = train_dkf(model, v, u, seed=SEED)
            r, cur = curve_rmse(model.do_query); r_ood = ood_rmse(model.do_query)
            # diagnostics: learned prior and encoder's correlation with the true latent
            with torch.no_grad():
                x = torch.stack([nv(torch.tensor(v[:, :-1], dtype=torch.float32)), nu(torch.tensor(u, dtype=torch.float32)), nv(torch.tensor(v[:, 1:], dtype=torch.float32))], -1)
                hdn, _ = model.enc(x); mu = model.enc_out(hdn)[..., 0]
            corr = float(np.corrcoef(mu.numpy().ravel(), c.ravel())[0, 1])
            res[name] = dict(rmse=r, rmse_ood=r_ood, curve=cur, loss=L, rho=float(torch.sigmoid(model.rho_raw)), s=float(torch.exp(model.log_s)),
                             corr_c=corr, dfdu=dfdu(model, v, u, mu))
        out[kind] = res
        print(f"[{kind}] seed={SEED} ({time.time()-t0:.0f}s)  " + "  ".join(f"{k}: {res[k]['rmse']:.2f} (ood {res[k]['rmse_ood']:.2f})" for k in res))
        for k in ['DKF', 'DKF-B']:
            print(f"    {k}: rho_hat={res[k]['rho']:.3f} s_hat={res[k]['s']:.3f} corr(enc,c)={res[k]['corr_c']:+.3f} dfdu={res[k]['dfdu']:.3f}")
        json.dump(out, open(os.path.join(HERE, f"exp2_results_seed{SEED}.json"), "w"), indent=1)
