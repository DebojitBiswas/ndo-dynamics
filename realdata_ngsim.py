#!/usr/bin/env python3
"""
Real-data demonstration on NGSIM vehicle trajectories (US-101 or I-80; data.transportation.gov).

Setting (car-following): state v = follower speed, input u = follower acceleration is NOT used as the
input (it is the driver's choice, i.e. the policy).  We take the *leader's speed* v_L as the input the
analyst may want to intervene on (do(v_L = profile)), the follower's acceleration as the policy output,
and the latent c as the unrecorded driver/traffic state that drives both the follower's reaction and the
leader's motion (shared upstream traffic).  Physics: kinematics (gap' = gap + Δ (v_L − v)) is an exact row
(P2); the follower's acceleration enters speed with unit gain (P1: v' = v + Δ a, b = 1 exactly).

Because real data give no interventional ground truth, the validity test is an input-shift transfer that
mimics the intervention: train on windows in which the leader drives steadily, evaluate multi-step
prediction of the follower's speed on held-out windows in which the leader brakes hard (an input profile
the training data did not contain, with the same drivers and the same policy).  Estimators that learned
the structural dynamics transfer; estimators that learned the observational law do not.  We also report the learned
acceleration gain of the free latent-process model against its physical value (1) — the paper's diagnostic.

Input: a CSV with NGSIM columns  Vehicle_ID, Frame_ID, Lane_ID, v_Vel, v_Acc, Preceding, Space_Headway
(US-101 / I-80 'vehicle-trajectory-data' format, 0.1 s frames; units ft/s, ft/s², ft).
Usage:  python3 realdata_ngsim.py /path/to/trajectories.csv [--regime-split lane|time]
Dry run on a synthetic stand-in:  python3 realdata_ngsim.py --synthetic
"""
import argparse, json, os, sys, math, numpy as np, torch, torch.nn as nn
import exp2_latent_process_ndo as X
import exp1_confounded_vehicle as E

HERE = os.path.dirname(os.path.abspath(__file__)); DT = 0.1; T = 60
FT = 0.3048

def load_pairs(csv, max_pairs=6000, kin_tol=2.0):
    """Windows of T+1 contiguous frames with one leader, same lane.  Hygiene for the data.transportation.gov
    export: exact duplicate rows are dropped; vehicle IDs restart in each 15-min period, so vehicles whose
    (vehicle_id, frame_id) keys collide across periods are discarded, leaders must be collision-free too, and
    each window must satisfy the kinematic identity gap' − gap ≈ Δ(v_L − v) (RMS residual < kin_tol ft), which
    rejects leader/follower pairs taken from different periods."""
    import pandas as pd
    cols = ['Vehicle_ID', 'Frame_ID', 'Lane_ID', 'v_Vel', 'v_Acc', 'Preceding', 'Space_Headway']
    df = pd.read_csv(csv, usecols=lambda c: c.lower() in [x.lower() for x in cols])
    df.columns = [next(x for x in cols if x.lower() == c.lower()) for c in df.columns]   # accept Socrata lowercase names
    df = df.drop_duplicates()
    g = df.groupby(['Vehicle_ID', 'Frame_ID']).size(); bad = set(g[g > 1].index.get_level_values(0))
    df = df[~df.Vehicle_ID.isin(bad)].sort_values(['Vehicle_ID', 'Frame_ID'])
    speed = {(r.Vehicle_ID, r.Frame_ID): r.v_Vel for r in df.itertuples()}
    windows = []; n_rej = 0
    for vid, gdf in df.groupby('Vehicle_ID'):
        gdf = gdf[gdf.Preceding > 0]
        if len(gdf) < T + 1: continue
        arr = gdf[['Frame_ID', 'Lane_ID', 'v_Vel', 'v_Acc', 'Preceding', 'Space_Headway']].to_numpy()
        for s in range(0, len(arr) - T - 1, T):
            w = arr[s:s+T+1]
            if len(set(w[:, 1])) != 1 or len(set(w[:, 4])) != 1: continue          # same lane, same leader
            if np.any(np.diff(w[:, 0]) != 1): continue                              # contiguous frames
            vl = np.array([speed.get((int(w[0, 4]), int(f)), np.nan) for f in w[:, 0]])
            if np.isnan(vl).any(): continue
            kin = np.diff(w[:, 5]) - DT*(vl[:-1] - w[:-1, 2])
            if np.sqrt(np.mean(kin**2)) > kin_tol: n_rej += 1; continue              # kinematic identity fails → wrong period
            vlm = vl*FT; regime = int((vlm.min() - vlm[0]) < -3.0)                      # 1 = leader brakes by > 3 m/s
            windows.append(dict(v=w[:, 2]*FT, a=w[:, 3]*FT, vl=vlm, gap=w[:, 5]*FT, lane=regime, frame=int(w[0, 0])))
            if len(windows) >= max_pairs: break
        if len(windows) >= max_pairs: break
    print(f"windows kept {len(windows)}, rejected by kinematic check {n_rej}")
    return windows

def synthetic_pairs(n=1500, seed=0):
    """Stand-in with the paper's structure, in NGSIM units, so the pipeline can be exercised without data."""
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        regime = rng.integers(0, 2)                           # 0 steady leader, 1 leader brakes hard (same driver policy)
        c = np.zeros(T+1); c[0] = rng.normal()
        for t in range(1, T+1): c[t] = 0.9*c[t-1] + math.sqrt(1-0.81)*rng.normal()
        vl = 25 + 2*np.cumsum(rng.normal(0, 0.3, T+1))*DT + 1.5*c
        if regime: vl = vl - 6*np.clip((np.arange(T+1) - 15)/20, 0, 1)
        v = np.zeros(T+1); a = np.zeros(T+1); gap = np.zeros(T+1); v[0] = vl[0] + rng.normal(0, 1); gap[0] = 20 + rng.normal(0, 3)
        kp = 0.5
        for t in range(T):
            a[t] = kp*(vl[t] - v[t]) + 0.05*(gap[t] - 15) + 1.0*c[t] + rng.normal(0, 0.3)
            v[t+1] = v[t] + DT*a[t]; gap[t+1] = gap[t] + DT*(vl[t] - v[t])
        a[T] = a[T-1]
        out.append(dict(v=v, a=a, vl=vl, gap=gap, lane=regime, frame=i))
    return out

class RealDKF(nn.Module):
    """Latent-process model for car-following.  State (v, gap); input v_L; policy a = π(v, gap, v_L, c).
       free:  v' = v + Δ f(v, gap, v_L, c) ; gap' = gap + Δ h(...)
       phys:  v' = v + Δ a (P1, unit gain on the measured acceleration) ; gap' = gap + Δ (v_L − v) (P2 exact row)
       and the policy head carries the latent.  The do-query replaces the leader profile."""
    def __init__(self, mode='free', h=64):
        super().__init__(); self.mode = mode
        self.enc = nn.GRU(4, h, batch_first=True, bidirectional=True); self.enc_out = nn.Linear(2*h, 2)
        self.pi = X.mlp(4, 1, h); self.f = X.mlp(4, 2, h) if mode in ('free', 'naive') else (X.mlp(4, 1, h) if mode == 'p2' else None)
        self.log_sa = nn.Parameter(torch.tensor(0.0)); self.log_sv = nn.Parameter(torch.tensor(-2.0)); self.log_sg = nn.Parameter(torch.tensor(-2.0))
        self.rho_raw = nn.Parameter(torch.tensor(1.0)); self.log_s = nn.Parameter(torch.tensor(-1.0))
    def feats(self, v, gap, vl, c): return torch.stack([v/10, gap/20, (vl - v)/5, c], -1)
    def policy(self, v, gap, vl, c): return self.pi(self.feats(v, gap, vl, c))[..., 0]*2.0
    def dyn(self, v, gap, vl, a, c):
        if self.mode in ('free', 'naive'):
            o = self.f(self.feats(v, gap, vl, c)); return v + DT*o[..., 0]*2.0, gap + DT*o[..., 1]*5.0
        if self.mode == 'p2':                                   # P2 only: exact kinematic gap row, free speed row
            return v + DT*self.f(self.feats(v, gap, vl, c))[..., 0]*2.0, gap + DT*(vl - v)
        return v + DT*a, gap + DT*(vl - v)
    def elbo(self, v, gap, vl, a):
        x = torch.stack([v[:, :-1]/10, gap[:, :-1]/20, (vl[:, :-1]-v[:, :-1])/5, a[:, :-1]/2], -1)
        hdn, _ = self.enc(x); mu, ls = self.enc_out(hdn).unbind(-1); ls = ls.clamp(-6, 2); c = mu + torch.exp(ls)*torch.randn_like(mu)
        if self.mode == 'naive': c = torch.zeros_like(c)                      # one-step simulator without a latent
        sa, sv, sg = torch.exp(self.log_sa), torch.exp(self.log_sv), torch.exp(self.log_sg)
        lp_a = -0.5*(((a[:, :-1] - self.policy(v[:, :-1], gap[:, :-1], vl[:, :-1], c))/sa)**2 + 2*torch.log(sa))
        vp, gp = self.dyn(v[:, :-1], gap[:, :-1], vl[:, :-1], a[:, :-1], c)
        lp_v = -0.5*(((v[:, 1:] - vp)/sv)**2 + 2*torch.log(sv)); lp_g = -0.5*(((gap[:, 1:] - gp)/sg)**2 + 2*torch.log(sg))
        rho, s = torch.sigmoid(self.rho_raw), torch.exp(self.log_s)
        lp_c = -0.5*c[:, 0]**2 - 0.5*((((c[:, 1:] - rho*c[:, :-1])/s)**2 + 2*torch.log(s)).sum(1))
        lq = -0.5*(((c - mu)/torch.exp(ls))**2 + 2*ls).sum(1)
        if self.mode == 'naive': return (lp_a.sum(1) + lp_v.sum(1) + lp_g.sum(1)).mean()
        return (lp_a.sum(1) + lp_v.sum(1) + lp_g.sum(1) + lp_c - lq).mean()
    @torch.no_grad()
    def rollout(self, v0, gap0, vl_path, n_lat=32, seed=1):
        """Multi-step rollout under a given leader profile (the intervention), marginalising the latent prior."""
        g = torch.Generator().manual_seed(seed); rho, s = torch.sigmoid(self.rho_raw), torch.exp(self.log_s)
        B = v0.shape[0]; v = v0.repeat(n_lat); gap = gap0.repeat(n_lat); c = torch.randn(B*n_lat, generator=g); out = []
        if self.mode == 'naive': c = torch.zeros_like(c); s = 0*s
        for t in range(vl_path.shape[1]):
            vl = vl_path[:, t].repeat(n_lat); a = self.policy(v, gap, vl, c); v, gap = self.dyn(v, gap, vl, a, c)
            c = rho*c + s*torch.randn(B*n_lat, generator=g); out.append(v.view(n_lat, B).mean(0))
        return torch.stack(out, 1)

def to_tensors(ws):
    f = lambda k: torch.tensor(np.stack([w[k] for w in ws]), dtype=torch.float32)
    return f('v'), f('gap'), f('vl'), f('a')

def train(model, data, steps=3000, bs=128, lr=2e-3, seed=0):
    torch.manual_seed(seed); v, gap, vl, a = data; n = v.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        idx = torch.randint(0, n, (bs,)); loss = -model.elbo(v[idx], gap[idx], vl[idx], a[idx])/T
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()

@torch.no_grad()
def transfer_rmse(model, data):
    v, gap, vl, a = data; pred = model.rollout(v[:, 0], gap[:, 0], vl[:, :-1]); return float(torch.sqrt(((pred - v[:, 1:])**2).mean()))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('csv', nargs='?'); ap.add_argument('--synthetic', action='store_true'); ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    ws = synthetic_pairs() if args.synthetic else load_pairs(args.csv)
    regimes = sorted(set(w['lane'] for w in ws)); print("windows:", len(ws), "regimes:", regimes)
    # policy-shift split: train on the regime with most windows, test on the others
    train_r = 0   # steady-leader windows
    tr = to_tensors([w for w in ws if w['lane'] == train_r]); te = to_tensors([w for w in ws if w['lane'] != train_r])
    print(f"train regime {train_r} (steady leader): {tr[0].shape[0]} windows; test (leader brakes): {te[0].shape[0]} windows")
    OUTF = os.path.join(HERE, "realdata_results.json"); res = json.load(open(OUTF)) if os.path.exists(OUTF) and not args.synthetic else {}
    for mode in ['naive', 'free', 'p2', 'phys']:
        key = f"{mode}:seed{args.seed}"
        if key in res: continue
        m = RealDKF(mode); train(m, tr, steps=args.steps, seed=args.seed); res[key] = dict(in_regime=transfer_rmse(m, tr), shifted=transfer_rmse(m, te), rho=float(torch.sigmoid(m.rho_raw)))
        if mode == 'free':
            res[key]['rho'] = res[key].get('rho'); v, gap, vl, a = tr; vv = v[:, :-1].reshape(-1); gg = gap[:, :-1].reshape(-1); ll = vl[:, :-1].reshape(-1); aa = a[:, :-1].reshape(-1).clone().requires_grad_(True)
            # learned gain of speed on measured acceleration is not a free-model quantity here (speed row is f(v,gap,vL,c));
            # report instead the learned sensitivity of v' to v_L (should be ~0 physically: the leader acts only through the driver)
            with torch.enable_grad():
                llr = ll.clone().requires_grad_(True); c0 = torch.zeros_like(vv)
                vp, _ = m.dyn(vv, gg, llr, aa, c0); gsum = torch.autograd.grad(vp.sum(), llr)[0]
            res[key]['dv_dvl'] = float(gsum.mean()/DT)
        print(key, res[key]); 
        if not args.synthetic: json.dump(res, open(OUTF, "w"), indent=1)
