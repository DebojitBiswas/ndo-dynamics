#!/usr/bin/env python3
"""
Experiment 5 — a two-state system (position p, speed v) with an exact kinematic row.

    p' = p + Δ v                                  (kinematics: exact, contains no latent)
    v' = v + Δ( b u − k v + g c ) + w             (as before; AR(1) latent ρ = 0.9, d = 1)
Query: E[p_T | do(u = ū), p_0 = 0, v_0 = v*] and E[v_T | ...], in-support ū ∈ [3,6] and out-of-support ū ∈ [7,9].

Estimators: NAIVE-2 (MLP one-step on (p,v,u) → (p',v')), DKF-2 (latent-process model with a free 2-row
dynamics network), DKF-2-P2 (kinematic row exact, v-row grey-box with b learned), DKF-2-P12 (… with b fixed).
Tests the claim that the picture survives a multi-state system and that a known exact row is harmless/helpful.
"""
import json, math, os, sys, time, numpy as np, torch, torch.nn as nn
import exp1_confounded_vehicle as E
import exp2_latent_process_ndo as X

torch.set_num_threads(int(os.environ.get('EXP2_THREADS', 1)))
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "exp5_results.json")
P = E.P; DT = P['dt']; T = P['T']; SEED = int(os.environ.get('EXP2_SEED', 11)); RHO = 0.9
U_IN, U_OUT = X.U_GRID, X.U_OOD

def simulate(n, seed):
    rng = np.random.default_rng(seed); c = X.latent_paths('AR1', n, rng)
    v = np.zeros((n, T+1)); p = np.zeros((n, T+1)); u = np.zeros((n, T)); v[:, 0] = P['v_star'] + rng.normal(0, 1, n)
    for t in range(T):
        u[:, t] = P['kp']*(P['v_star'] - v[:, t]) + c[:, t] + rng.normal(0, P['sig_n'], n)
        p[:, t+1] = p[:, t] + DT*v[:, t]
        v[:, t+1] = v[:, t] + DT*(P['b']*u[:, t] - P['k']*v[:, t] + P['g']*c[:, t]) + rng.normal(0, P['sig_w'], n)
    return p, v, u, c

def gt(u_bar):
    p, v = 0.0, P['v_star']
    for _ in range(T): p, v = p + DT*v, v + DT*(P['b']*u_bar - P['k']*v)
    return p, v
GT_IN = np.array([gt(ub) for ub in U_IN]); GT_OUT = np.array([gt(ub) for ub in U_OUT])

def np_(p): return p/100.0
def nv(v): return (v - P['v_star'])/5.0
def nu(u): return u/2.0

class DKF2(nn.Module):
    def __init__(self, mode='free', known_b=None, h=64):
        super().__init__(); self.mode, self.known_b = mode, known_b
        self.enc = nn.GRU(5, h, batch_first=True, bidirectional=True); self.enc_out = nn.Linear(2*h, 2)
        self.pi = X.mlp(2, 1, h)
        if mode == 'free': self.f = X.mlp(4, 2, h)
        elif mode == 'kin': self.f = X.mlp(4, 1, h)                      # kinematic row exact, speed row free
        else:
            self.b_raw = nn.Parameter(torch.tensor(0.5)); self.k_raw = nn.Parameter(torch.tensor(0.1)); self.k2_raw = nn.Parameter(torch.tensor(0.0)); self.g_raw = nn.Parameter(torch.tensor(0.5))
        self.log_sn = nn.Parameter(torch.tensor(0.0)); self.log_sw = nn.Parameter(torch.tensor(-2.0)); self.log_sp = nn.Parameter(torch.tensor(-4.0))
        self.rho_raw = nn.Parameter(torch.tensor(1.0)); self.log_s = nn.Parameter(torch.tensor(-1.0))
    def dyn(self, p, v, u, c):
        if self.mode == 'free':
            o = self.f(torch.stack([np_(p), nv(v), nu(u), c], -1)); return p + DT*o[..., 0]*5.0, v + DT*o[..., 1]*US
        if self.mode == 'kin':
            return p + DT*v, v + DT*self.f(torch.stack([np_(p), nv(v), nu(u), c], -1))[..., 0]*US
        b = self.b_raw if self.known_b is None else torch.as_tensor(float(self.known_b))
        return p + DT*v, v + DT*(b*u - self.k_raw*v - self.k2_raw*v*torch.abs(v)/P['v_star'] + self.g_raw*c)
    def elbo(self, p, v, u):
        x = torch.stack([np_(p[:, :-1]), nv(v[:, :-1]), nu(u), np_(p[:, 1:]), nv(v[:, 1:])], -1)
        hdn, _ = self.enc(x); mu, ls = self.enc_out(hdn).unbind(-1); ls = ls.clamp(-6, 2); c = mu + torch.exp(ls)*torch.randn_like(mu)
        sn, sw, sp = torch.exp(self.log_sn), torch.exp(self.log_sw), torch.exp(self.log_sp)
        pp, vp = self.dyn(p[:, :-1], v[:, :-1], u, c)
        lp_u = -0.5*(((u - self.pi(torch.stack([nv(v[:, :-1]), c], -1))[..., 0]*US)/sn)**2 + 2*torch.log(sn))
        lp_v = -0.5*(((v[:, 1:] - vp)/sw)**2 + 2*torch.log(sw)); lp_p = -0.5*(((p[:, 1:] - pp)/sp)**2 + 2*torch.log(sp))
        rho, s = torch.sigmoid(self.rho_raw), torch.exp(self.log_s)
        lp_c = -0.5*(c[:, 0]**2) - 0.5*((((c[:, 1:] - rho*c[:, :-1])/s)**2 + 2*torch.log(s)).sum(1))
        lq = -0.5*(((c - mu)/torch.exp(ls))**2 + 2*ls).sum(1)
        return (lp_u.sum(1) + lp_v.sum(1) + lp_p.sum(1) + lp_c - lq).mean()
    @torch.no_grad()
    def do_query(self, u_bar, n=4000, seed=1):
        g = torch.Generator().manual_seed(seed); rho, s = torch.sigmoid(self.rho_raw), torch.exp(self.log_s)
        c = torch.randn(n, generator=g); p = torch.zeros(n); v = torch.full((n,), float(P['v_star'])); u = torch.full((n,), float(u_bar))
        for t in range(T):
            p, v = self.dyn(p, v, u, c); c = rho*c + s*torch.randn(n, generator=g)
        return float(p.mean()), float(v.mean())
US = 2.0

class Naive2(nn.Module):
    def __init__(self, h=64): super().__init__(); self.f = X.mlp(3, 2, h)
    def step(self, p, v, u): o = self.f(torch.stack([np_(p), nv(v), nu(u)], -1)); return p + DT*o[..., 0]*5.0, v + DT*o[..., 1]*US
    @torch.no_grad()
    def do_query(self, u_bar, n=1):
        p, v = torch.zeros(1), torch.full((1,), float(P['v_star'])); u = torch.full((1,), float(u_bar))
        for t in range(T): p, v = self.step(p, v, u)
        return float(p), float(v)

def train(model, p, v, u, steps=4000, bs=256, lr=2e-3, naive=False):
    torch.manual_seed(SEED); p = torch.tensor(p, dtype=torch.float32); v = torch.tensor(v, dtype=torch.float32); u = torch.tensor(u, dtype=torch.float32); n = p.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        idx = torch.randint(0, n, (bs,))
        if naive:
            pp, vp = model.step(p[idx, :-1], v[idx, :-1], u[idx]); loss = (((pp - p[idx, 1:])/0.005)**2).mean() + (((vp - v[idx, 1:])/P['sig_w'])**2).mean()
        else: loss = -model.elbo(p[idx], v[idx], u[idx])/T
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()

def rmse(model):
    ci = np.array([model.do_query(ub) for ub in U_IN]); co = np.array([model.do_query(ub) for ub in U_OUT])
    return dict(p_in=float(np.sqrt(np.mean((ci[:, 0]-GT_IN[:, 0])**2))), v_in=float(np.sqrt(np.mean((ci[:, 1]-GT_IN[:, 1])**2))),
                p_out=float(np.sqrt(np.mean((co[:, 0]-GT_OUT[:, 0])**2))), v_out=float(np.sqrt(np.mean((co[:, 1]-GT_OUT[:, 1])**2))))

if __name__ == "__main__":
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    p, v, u, c = simulate(1500, SEED); print("GT p_T in-support:", np.round(GT_IN[:, 0], 1))
    for name, ctor in [('NAIVE-2', lambda: Naive2()), ('DKF-2', lambda: DKF2('free')), ('DKF-2-KIN', lambda: DKF2('kin')), ('DKF-2-P2', lambda: DKF2('grey')), ('DKF-2-P12', lambda: DKF2('grey', P['b']))]:
        key = f"{name}:seed{SEED}"
        if key in res: continue
        m = ctor(); t0 = time.time(); train(m, p, v, u, naive=(name == 'NAIVE-2')); r = rmse(m)
        if name.startswith('DKF-2-P'): r['coeffs'] = dict(b=float(m.b_raw) if m.known_b is None else 1.0, k=float(m.k_raw), k2=float(m.k2_raw))
        res[key] = r; json.dump(res, open(OUT, "w"), indent=1)
        print(f"[{key}] ({time.time()-t0:.0f}s) p: {r['p_in']:.2f}/{r['p_out']:.2f}  v: {r['v_in']:.2f}/{r['v_out']:.2f}", r.get('coeffs', ''))
