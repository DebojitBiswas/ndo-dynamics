#!/usr/bin/env python3
"""
Kalman maximum-likelihood estimator for class C0 — the estimator Proposition 4 says should work.

Likelihood of one trajectory (v_0..v_T, u_0..u_{T-1}) with the latent c_t marginalised by a scalar
Kalman filter.  Conditioning on the observed v's, each step contributes two Gaussian observations of c_t:

    u_t     | c_t  ~ N( kp (v* - v_t) + d c_t ,  σ_n² )                    (policy = noisy measurement of c)
    v_{t+1} | c_t  ~ N( v_t + Δ(b u_t - k v_t) + Δ g c_t ,  σ_w² )         (dynamics = second measurement)
    c_{t+1} | c_t  ~ N( ρ c_t , (1-ρ²) σ_c² ),  σ_c := 1,  c_0 ~ N(0,1)

θ = (b, k, g, kp, v*, d, ρ, σ_n, σ_w); with physics P1, b is fixed at its true value.
Fit by L-BFGS from deliberately wrong initial values; standard errors from the inverse observed
information (Hessian of the negative log-likelihood at the optimum).  Outputs theory_kalman_ml.json.
"""
import json, os, math, time, numpy as np, torch
import exp1_confounded_vehicle as E

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
P = E.P; DT = P['dt']
NAMES = ['b', 'k', 'g', 'kp', 'v_star', 'd', 'rho', 'sig_n', 'sig_w']

def unpack(raw, b_known):
    """raw: unconstrained tensor -> dict of constrained params."""
    i = 0; th = {}
    if b_known is None:
        th['b'] = raw[i]; i += 1
    else:
        th['b'] = torch.as_tensor(float(b_known))
    th['k'] = raw[i]; th['g'] = raw[i+1]; th['kp'] = raw[i+2]; th['v_star'] = raw[i+3]; th['d'] = raw[i+4]
    th['rho'] = torch.sigmoid(raw[i+5]); th['sig_n'] = torch.exp(raw[i+6]); th['sig_w'] = torch.exp(raw[i+7])
    return th

def pack(th, b_known):
    vals = ([] if b_known is not None else [th['b']]) + [th['k'], th['g'], th['kp'], th['v_star'], th['d'],
            math.log(th['rho']/(1-th['rho'])), math.log(th['sig_n']), math.log(th['sig_w'])]
    return torch.tensor(vals, requires_grad=True)

def negloglik(raw, v, u, b_known):
    th = unpack(raw, b_known)
    b, k, g, kp, vs, d, rho, sn, sw = [th[n] for n in NAMES]
    N, T = u.shape
    m = torch.zeros(N); Pc = torch.ones(N); ll = torch.zeros(N)
    log2pi = math.log(2*math.pi)
    for t in range(T):
        # observe u_t
        nu = u[:, t] - (kp*(vs - v[:, t]) + d*m); S = d*d*Pc + sn*sn
        ll = ll - 0.5*(log2pi + torch.log(S) + nu*nu/S)
        K = d*Pc/S; m = m + K*nu; Pc = Pc - K*d*Pc
        # observe v_{t+1}
        H = DT*g
        nu = v[:, t+1] - (v[:, t] + DT*(b*u[:, t] - k*v[:, t]) + H*m); S = H*H*Pc + sw*sw
        ll = ll - 0.5*(log2pi + torch.log(S) + nu*nu/S)
        K = H*Pc/S; m = m + K*nu; Pc = Pc - K*H*Pc
        # predict
        m = rho*m; Pc = rho*rho*Pc + (1 - rho*rho)
    return -ll.sum()

def fit(v, u, b_known=None, inits=None, iters=200):
    v = torch.tensor(v); u = torch.tensor(u)
    best = None
    for init in inits:
        raw = pack(init, b_known)
        opt = torch.optim.LBFGS([raw], lr=0.5, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                                history_size=50, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(); L = negloglik(raw, v, u, b_known); L.backward(); return L
        opt.step(closure)
        L = negloglik(raw, v, u, b_known).item()
        if best is None or L < best[0]:
            best = (L, raw.detach().clone())
    L, raw = best
    # observed information -> standard errors in the constrained parameterisation (delta method via autograd)
    def nll_c(raw_): return negloglik(raw_, v, u, b_known)
    Hs = torch.autograd.functional.hessian(nll_c, raw)
    cov_raw = torch.linalg.inv(Hs)
    th = unpack(raw, b_known)
    # Jacobian of constrained params wrt raw
    def constrained(raw_):
        t_ = unpack(raw_, b_known); return torch.stack([t_[n] for n in NAMES if not (b_known is not None and n == 'b')])
    Jc = torch.autograd.functional.jacobian(constrained, raw)
    cov = Jc @ cov_raw @ Jc.T
    se = torch.sqrt(torch.clamp(torch.diag(cov), min=0))
    names = [n for n in NAMES if not (b_known is not None and n == 'b')]
    est = {n: float(th[n]) for n in NAMES}
    ses = {n: float(s) for n, s in zip(names, se)}
    return dict(nll=L, est=est, se=ses)

def do_rmse(b_hat, k_hat):
    GT = np.array([E.gt_do(ub) for ub in np.linspace(3, 6, 7)])
    out = []
    for ub in np.linspace(3, 6, 7):
        x = P['v_star']
        for _ in range(P['T']): x = x + DT*(b_hat*ub - k_hat*x)
        out.append(x)
    return float(np.sqrt(np.mean((np.array(out) - GT)**2)))

if __name__ == "__main__":
    N = 1000
    # deliberately wrong starting points (two of them)
    inits = [dict(b=1.6, k=0.05, g=0.5, kp=0.5, v_star=18.0, d=0.3, rho=0.5, sig_n=1.0, sig_w=0.2),
             dict(b=0.7, k=0.6, g=1.5, kp=1.2, v_star=22.0, d=1.5, rho=0.8, sig_n=0.3, sig_w=0.05)]
    results = []
    print(f"Kalman-ML, N={N}, T={P['T']}, d=1.  truth: b=1, k=0.3, g=1, kp=0.8, v*=20, d=1, sig_n=0.5, sig_w=0.05")
    print(" ρ    | b free:  b̂ (se)      k̂ (se)      ρ̂     do-RMSE || b known:  k̂ (se)      ρ̂     do-RMSE | KB-FE RMSE")
    KBFE = {r['rho']: r['KB-FE']['rmse'] for r in json.load(open(os.path.join(HERE, 'theory_estimators.json')))['panelB']}
    for rho in [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]:
        v, u, c = E.simulate(1.0, rho, n_traj=N, seed=7)
        t0 = time.time()
        F = fit(v, u, None, inits); Fk = fit(v, u, P['b'], inits)
        rF = do_rmse(F['est']['b'], F['est']['k']); rK = do_rmse(P['b'], Fk['est']['k'])
        results.append(dict(rho=rho, free=F, known_b=Fk, rmse_free=rF, rmse_known=rK, secs=time.time()-t0))
        print(f" {rho:4.2f} |  {F['est']['b']:.3f} ({F['se']['b']:.3f})  {F['est']['k']:.3f} ({F['se']['k']:.3f})  {F['est']['rho']:.3f}   {rF:5.2f}  ||"
              f"  {Fk['est']['k']:.3f} ({Fk['se']['k']:.3f})  {Fk['est']['rho']:.3f}   {rK:5.2f}  |  {KBFE.get(rho, float('nan')):5.2f}")
    json.dump(results, open(os.path.join(HERE, "theory_kalman_ml.json"), "w"), indent=1)
