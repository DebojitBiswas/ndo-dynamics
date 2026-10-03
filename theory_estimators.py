#!/usr/bin/env python3
"""
Linear analogues of the three neural estimators in exp1, computed by exact least squares on a
large simulated panel (N = 20000 trajectories, T = 60) so sampling error is negligible and the
neural approximation is taken out of the picture.  Whatever bias remains is a property of the
*estimator*, not of the network.

  OLS      : v' on (1, v, u)                                  -> naive one-step simulator
  FE       : within-trajectory demeaned v' on (v, u)           -> NDO-Z  (per-trajectory latent)
  KB       : r := v' - v - Δ b u  on (1, v)                    -> known input gain, no latent
  KB-FE    : demeaned r on demeaned v                          -> NDO-ZB (known b + per-trajectory latent)
  KB-FD    : first-differenced r on first-differenced v        -> known b + differencing (Arellano-Bond flavour, no IV)

Do-curve: E[v_T | do(u = ū), v_0 = v*] from the fitted linear law with E[c] = 0, compared with truth.
Outputs theory_estimators.json and prints the table.
"""
import json, os, numpy as np
import exp1_confounded_vehicle as E

HERE = os.path.dirname(os.path.abspath(__file__))
P = E.P; dt = P['dt']; U_GRID = np.linspace(3.0, 6.0, 7)
GT = np.array([E.gt_do(ub) for ub in U_GRID])

def do_curve(b_hat, phi_v, c0=0.0):
    out = []
    for ub in U_GRID:
        v = P['v_star']
        for _ in range(P['T']):
            v = v + dt*(b_hat*ub + phi_v*v + c0)
        out.append(v)
    return np.array(out)

def rmse(curve): return float(np.sqrt(np.mean((curve - GT)**2)))

def fit_all(v, u, b_true=P['b']):
    V, Vn, U = v[:, :-1], v[:, 1:], u
    dv = (Vn - V)/dt                                   # observed acceleration
    res = {}
    # OLS
    X = np.stack([np.ones(V.size), V.ravel(), U.ravel()], 1); beta = np.linalg.lstsq(X, dv.ravel(), rcond=None)[0]
    res['OLS'] = dict(b=beta[2], phi=beta[1], c0=beta[0] - 0.0)
    # FE (within)
    dm = lambda M: M - M.mean(1, keepdims=True)
    X = np.stack([dm(V).ravel(), dm(U).ravel()], 1); beta = np.linalg.lstsq(X, dm(dv).ravel(), rcond=None)[0]
    # intercept recovered from grand means so that the rollout has the right level
    c0 = dv.mean() - beta[0]*V.mean() - beta[1]*U.mean()
    res['FE'] = dict(b=beta[1], phi=beta[0], c0=c0)
    # KB: known b
    r = dv - b_true*U
    X = np.stack([np.ones(V.size), V.ravel()], 1); beta = np.linalg.lstsq(X, r.ravel(), rcond=None)[0]
    res['KB'] = dict(b=b_true, phi=beta[1], c0=beta[0])
    # KB-FE
    X = dm(V).ravel()[:, None]; beta = np.linalg.lstsq(X, dm(r).ravel(), rcond=None)[0]
    res['KB-FE'] = dict(b=b_true, phi=beta[0], c0=r.mean() - beta[0]*V.mean())
    # KB-FD
    X = np.diff(V, axis=1).ravel()[:, None]; beta = np.linalg.lstsq(X, np.diff(r, axis=1).ravel(), rcond=None)[0]
    res['KB-FD'] = dict(b=b_true, phi=beta[0], c0=r.mean() - beta[0]*V.mean())
    for k in res:
        # the intercept absorbs Δ kp b v*-type constants only in the closed-loop form; under do(u) the true
        # intercept is 0 (E[c]=0), so we evaluate the rollout with the fitted c0 *and* report it.
        res[k]['rmse'] = rmse(do_curve(res[k]['b'], res[k]['phi'], res[k]['c0']))
        res[k]['rmse_c0_zero'] = rmse(do_curve(res[k]['b'], res[k]['phi'], 0.0))
    return res

if __name__ == "__main__":
    N = 20000
    out = dict(panelA=[], panelB=[])
    print("true: b = %.2f, phi_v = -k = %.2f, c0 = 0\n" % (P['b'], -P['k']))
    hdr = " %-6s| " + " | ".join("%-22s" % k for k in ['OLS','FE','KB','KB-FE','KB-FD'])
    print("Panel A: ρ = 1, sweep d.   cells: b̂ / φ̂_v / do-RMSE")
    print(hdr % "d")
    for d in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        v, u, c = E.simulate(d, 1.0, n_traj=N, seed=0); R = fit_all(v, u); R['d'] = d; out['panelA'].append(R)
        print((" %-6.2f| " % d) + " | ".join("%5.3f/%6.3f/%6.2f" % (R[k]['b'], R[k]['phi'], R[k]['rmse']) for k in ['OLS','FE','KB','KB-FE','KB-FD']))
    print("\nPanel B: d = 1, sweep ρ.")
    print(hdr % "ρ")
    for rho in [1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]:
        v, u, c = E.simulate(1.0, rho, n_traj=N, seed=0); R = fit_all(v, u); R['rho'] = rho; out['panelB'].append(R)
        print((" %-6.2f| " % rho) + " | ".join("%5.3f/%6.3f/%6.2f" % (R[k]['b'], R[k]['phi'], R[k]['rmse']) for k in ['OLS','FE','KB','KB-FE','KB-FD']))
    json.dump(out, open(os.path.join(HERE, "theory_estimators.json"), "w"), indent=1)
