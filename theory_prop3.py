#!/usr/bin/env python3
"""
Proposition 3 — exact population limit (N → ∞) of the known-b within (fixed-effects) estimator
under an AR(1) latent, finite T.  No simulation: the full joint covariance of the trajectory
(v_0..v_{T-1}, c_0..c_{T-1}) is built from the linear recursion and the within-estimator's
probability limit is a ratio of traces against the demeaning matrix M = I - 11ᵀ/T.

   φ̂_KB-FE  →  -k + g · tr(M K_cv) / tr(M K_vv)
with K_vv = Cov(v-vector) + m_v m_vᵀ (the deterministic transient mean path survives demeaning),
     K_cv = Cov(c-vector, v-vector)        (E c = 0).
Also the same object for the naive-OLS state coefficient (no demeaning, pooled over t) and the
known-b, no-latent estimator, for the full picture.
"""
import json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = dict(dt=0.1, T=60, b=1.0, k=0.3, g=1.0, kp=0.8, v_star=20.0, sig_c=1.0, sig_n=0.5, sig_w=0.05)

def trajectory_moments(P, d, rho):
    dt, b, k, g, kp, vs, T = P['dt'], P['b'], P['k'], P['g'], P['kp'], P['v_star'], P['T']
    sc2, sn2, sw2 = P['sig_c']**2, P['sig_n']**2, P['sig_w']**2
    a = 1 - dt*(k + b*kp)
    A = np.array([[a, dt*(b*d + g)], [0.0, rho]]); const = np.array([dt*kp*b*vs, 0.0])
    Q = np.diag([dt**2*b**2*sn2 + sw2, (1 - rho**2)*sc2])
    # per-time means and covariances, and cross-covariances Cov(s_t, s_t') = A^(t'-t) Σ_t
    m = [np.array([vs, 0.0])]; S = [np.diag([1.0, sc2])]
    for t in range(T - 1):
        m.append(A @ m[-1] + const); S.append(A @ S[-1] @ A.T + Q)
    K = np.zeros((2*T, 2*T))  # block (t, t') 2x2
    for t in range(T):
        Apow = np.eye(2)
        for tp in range(t, T):
            C = Apow @ S[t] if tp == t else Apow @ S[t]    # Cov(s_tp, s_t) = A^(tp-t) Σ_t
            K[2*tp:2*tp+2, 2*t:2*t+2] = C; K[2*t:2*t+2, 2*tp:2*tp+2] = C.T
            Apow = A @ Apow
    mv = np.array([mm[0] for mm in m])
    Kvv = K[0::2, 0::2]; Kcv = K[1::2, 0::2]; Kcc = K[1::2, 1::2]
    return mv, Kvv, Kcv, Kcc

def within_limit(P, d, rho):
    T = P['T']; M = np.eye(T) - np.ones((T, T))/T
    mv, Kvv, Kcv, Kcc = trajectory_moments(P, d, rho)
    # Nickell term: Cov(w_t, v_t') = a^(t'-t-1) sigma_w^2 for t' > t  (w_t enters v_{t+1})
    a = 1 - P['dt']*(P['k'] + P['b']*P['kp']); Kwv = np.zeros((T, T))
    for t in range(T):
        for tp in range(t + 1, T): Kwv[t, tp] = a**(tp - t - 1)*P['sig_w']**2
    num = np.trace(M @ Kcv); den = np.trace(M @ (Kvv + np.outer(mv, mv))); nick = np.trace(M @ Kwv)/P['dt']
    phi_fe = -P['k'] + (P['g']*num + nick)/den
    # known-b, no latent (pooled OLS of r on (1, v)): pooled over t, centre by grand mean
    mu_v = mv.mean(); Vv = (np.trace(Kvv) + np.sum((mv - mu_v)**2))/T; Cvc = np.trace(Kcv)/T
    phi_kb = -P['k'] + P['g']*Cvc/Vv
    return phi_fe, phi_kb, num/den

if __name__ == "__main__":
    S = json.load(open(os.path.join(HERE, "theory_estimators.json")))
    print("Known-b estimators, d = 1: exact population limit vs large-N least squares (true φ_v = -0.300)")
    print(" ρ     | KB-FE exact | KB-FE sim | KB exact | KB sim")
    rows = []
    for r in S['panelB']:
        rho = r['rho']
        if rho >= 1.0:
            continue
        fe, kb, ratio = within_limit(P, 1.0, rho)
        rows.append(dict(rho=rho, phi_fe=fe, phi_kb=kb))
        print(f" {rho:5.2f} |   {fe:+.4f}   |  {r['KB-FE']['phi']:+.4f}  |  {kb:+.4f} | {r['KB']['phi']:+.4f}")
    # fine sweep to locate the peak of the hump
    grid = np.linspace(0.0, 0.999, 200); vals = [within_limit(P, 1.0, x)[0] for x in grid]
    i = int(np.argmax(vals)); print(f"\nhump: max φ̂_KB-FE = {vals[i]:+.4f} at ρ = {grid[i]:.3f}  (bias {vals[i]+P['k']:+.4f})")
    json.dump(dict(rows=rows, grid=grid.tolist(), phi_fe=vals), open(os.path.join(HERE, "theory_prop3.json"), "w"))
