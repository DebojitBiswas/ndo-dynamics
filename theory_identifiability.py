#!/usr/bin/env python3
"""
Local identifiability of class C0 from the observational law of (v_t, u_t), stationary regime (ρ < 1).

Augmented state s_t = (v_t, c_t, n_t):
    v_{t+1} = a v_t + Δ(bd+g) c_t + Δ b n_t + Δ kp b v* + w_t
    c_{t+1} = ρ c_t + ξ_{t+1},            Var ξ = (1-ρ²) σ_c²,  σ_c := 1 (scale fixed; (g,d) carry the scale)
    n_{t+1} = η_{t+1},                    Var η = σ_n²
Observation y_t = (v_t, u_t) = H s_t + h0,  H = [[1,0,0],[-kp, d, 1]],  h0 = (0, kp v*).

Observational law (Gaussian, stationary) is determined by  mean(y), Γ(0), Γ(1), ..., Γ(L).
θ = (b, k, g, kp, d, ρ, σ_n, σ_w, v*).  Local identifiability at θ0 ⇔ Jacobian of the map
θ ↦ [mean, Γ(0..L)] has full column rank.  We compute the Jacobian by central differences and
report singular values, with and without b held fixed (physics P1), across (d, ρ).
The smallest right-singular vector shows which parameter combination is unidentified.
"""
import numpy as np, itertools, json, os
from scipy.linalg import solve_discrete_lyapunov

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['b', 'k', 'g', 'kp', 'd', 'rho', 'sig_n', 'sig_w', 'v_star']
DT = 0.1; L = 12

def moments(theta, L=L):
    b, k, g, kp, d, rho, sn, sw, vs = theta
    a = 1 - DT*(k + b*kp)
    A = np.array([[a, DT*(b*d + g), DT*b], [0, rho, 0], [0, 0, 0]])
    const = np.array([DT*kp*b*vs, 0, 0])
    Q = np.diag([sw**2, (1 - rho**2), sn**2])
    H = np.array([[1, 0, 0], [-kp, d, 1.0]]); h0 = np.array([0, kp*vs])
    m = np.linalg.solve(np.eye(3) - A, const)
    S = solve_discrete_lyapunov(A, Q)
    out = [H @ m + h0]
    Al = np.eye(3)
    for l in range(L + 1):
        G = H @ Al @ S @ H.T
        out.append(G.ravel() if l > 0 else G[np.triu_indices(2)])
        Al = Al @ A
    return np.concatenate(out)

def jacobian(theta, free, h=1e-6):
    theta = np.array(theta, float); cols = []
    for i in free:
        tp = theta.copy(); tm = theta.copy(); tp[i] += h; tm[i] -= h
        cols.append((moments(tp) - moments(tm))/(2*h))
    return np.stack(cols, 1)

def analyze(theta, fix_b):
    free = [i for i in range(len(NAMES)) if not (fix_b and NAMES[i] == 'b')]
    J = jacobian(theta, free)
    # scale columns by parameter magnitude for a scale-free rank test
    scale = np.array([max(abs(theta[i]), 1.0) for i in free]); Js = J*scale
    U, s, Vt = np.linalg.svd(Js, full_matrices=False)
    null = Vt[-1]
    return s, [(NAMES[free[i]], round(float(null[i]), 3)) for i in np.argsort(-np.abs(null))[:4]]

if __name__ == "__main__":
    base = dict(b=1.0, k=0.3, g=1.0, kp=0.8, sig_n=0.5, sig_w=0.05, v_star=20.0)
    results = []
    print("singular values of the (scaled) Jacobian of θ ↦ observational moments; ratio = s_min / s_max")
    print(" d     ρ    | b free: s_min/s_max   null direction                 | b known: s_min/s_max")
    for d, rho in itertools.product([0.0, 0.5, 1.0, 2.0], [0.0, 0.5, 0.9, 0.99]):
        th = [base['b'], base['k'], base['g'], base['kp'], d, rho, base['sig_n'], base['sig_w'], base['v_star']]
        s_free, null = analyze(th, fix_b=False); s_fix, _ = analyze(th, fix_b=True)
        r_free = s_free[-1]/s_free[0]; r_fix = s_fix[-1]/s_fix[0]
        results.append(dict(d=d, rho=rho, ratio_b_free=float(r_free), ratio_b_known=float(r_fix), null=null))
        print(f" {d:4.1f}  {rho:4.2f} |   {r_free:9.2e}        {str(null):45s} |   {r_fix:9.2e}")
    json.dump(results, open(os.path.join(HERE, "theory_identifiability.json"), "w"), indent=1)
