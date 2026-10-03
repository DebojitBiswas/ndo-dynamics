#!/usr/bin/env python3
"""
Lemma 1 — exact population bias of the one-step regression, class C0.

Model (dt = Δ):
    v_{t+1} = v_t + Δ( b u_t - k v_t + g c_t ) + w_t
    u_t     = kp (v* - v_t) + d c_t + n_t
    c_t     = ρ c_{t-1} + sqrt(1-ρ²) ξ_t,   stationary Var(c_t) = σ_c²   (ρ = 1: constant per trajectory)

Substituting the policy into the dynamics gives the closed loop
    v_{t+1} = a v_t + Δ kp b v* + Δ (b d + g) c_t + Δ b n_t + w_t,      a := 1 - Δ (k + b kp)

Population one-step OLS of v_{t+1} on (1, v_t, u_t):
    β = β_true + Δ g γ,   γ := Σ_XX^{-1} Σ_Xc,  X = (v_t, u_t) centered.
    => b̂ = b + g γ_u ,   k̂ = k - g γ_v         (after dividing by Δ)

Two ways to get Σ:
  (A) stationary closed form (ρ = 1): derived by hand, see theory_v0.md.
  (B) exact finite-T moment recursion for the experiment's actual design
      (v_0 ~ N(v*, 1), T = 60, pooled over t): propagate the joint second moments
      of (v_t, c_t) step by step — no simulation, no sampling error.
Both are compared with the fitted OLS from exp1_results.json.
"""
import json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = dict(dt=0.1, T=60, b=1.0, k=0.3, g=1.0, kp=0.8, v_star=20.0, sig_c=1.0, sig_n=0.5, sig_w=0.05)

def closed_loop_a(P):
    return 1 - P['dt']*(P['k'] + P['b']*P['kp'])

def stationary_cov_rho1(P, d):
    """(A) ρ = 1: c constant per trajectory. Returns Σ of (v, u, c) and γ."""
    dt, b, k, g, kp, sc2, sn2, sw2 = P['dt'], P['b'], P['k'], P['g'], P['kp'], P['sig_c']**2, P['sig_n']**2, P['sig_w']**2
    a = closed_loop_a(P)
    # given c, the trajectory is AR(1) around mean  m(c) = Δ(kp b v* + (bd+g) c)/(1-a)
    slope_vc = dt*(b*d + g)/(1 - a)                       # dm/dc
    within   = (dt**2*b**2*sn2 + sw2)/(1 - a**2)          # stationary within-trajectory variance
    Vv  = slope_vc**2*sc2 + within
    Cvc = slope_vc*sc2
    Cvu = -kp*Vv + d*Cvc                                   # Cov(v_t, n_t)=0
    Vu  = kp**2*Vv + d*d*sc2 + sn2 - 2*kp*d*Cvc
    Cuc = -kp*Cvc + d*sc2
    S_XX = np.array([[Vv, Cvu],[Cvu, Vu]]); S_Xc = np.array([Cvc, Cuc])
    gam = np.linalg.solve(S_XX, S_Xc)
    # phi_v := fitted coefficient on v_t, divided by Δ, minus the '1' = -k + g*gamma_v  (true value -k)
    return dict(gamma_v=gam[0], gamma_u=gam[1], b_hat=b + g*gam[1], phi_v=-k + g*gam[0], Vv=Vv, Vu=Vu, Cvu=Cvu, Cvc=Cvc, Cuc=Cuc)

def finite_T_moments(P, d, rho=1.0):
    """(B) exact pooled second moments of (v_t, u_t, c_t) over t = 0..T-1 for the experiment's design."""
    dt, b, k, g, kp, vs = P['dt'], P['b'], P['k'], P['g'], P['kp'], P['v_star']
    sc2, sn2, sw2 = P['sig_c']**2, P['sig_n']**2, P['sig_w']**2
    a = closed_loop_a(P)
    # state s = (v, c); mean m, covariance C.  c_0 ~ N(0, sc2) independent of v_0 ~ N(v*, 1)
    m = np.array([vs, 0.0]); C = np.diag([1.0, sc2])
    A = np.array([[a, dt*(b*d + g)], [0.0, rho]])
    const = np.array([dt*kp*b*vs, 0.0])
    Q = np.diag([dt**2*b**2*sn2 + sw2, (1 - rho**2)*sc2])
    # pooled accumulators for (v, u, c): u = kp(vs - v) + d c + n
    L = np.array([[1, 0], [-kp, d], [0, 1]], float); off = np.array([0.0, kp*vs, 0.0])
    S1 = np.zeros(3); S2 = np.zeros((3, 3)); n = 0
    for t in range(P['T']):
        mu = L @ m + off
        Cov = L @ C @ L.T; Cov[1, 1] += sn2
        S1 += mu; S2 += Cov + np.outer(mu, mu); n += 1
        m = A @ m + const; C = A @ C @ A.T + Q
    mean = S1/n; Sig = S2/n - np.outer(mean, mean)
    S_XX = Sig[:2, :2]; S_Xc = Sig[:2, 2]
    gam = np.linalg.solve(S_XX, S_Xc)
    return dict(gamma_v=gam[0], gamma_u=gam[1], b_hat=b + g*gam[1], phi_v=-k + g*gam[0])

if __name__ == "__main__":
    R = json.load(open(os.path.join(HERE, "exp1_results.json")))
    print("Lemma 1 check, ρ = 1 (constant latent)")
    print(" (true b = 1.0, true φ_v = -k = -0.3;  'stationary' = long-run regime, 'finite-T' = the experiment's transient design)")
    print(" d    | fitted b̂ | finite-T exact | stationary closed form || fitted φ_v | finite-T | stationary")
    for r in R["panelA"]:
        d = r["d"]; A = stationary_cov_rho1(P, d); B = finite_T_moments(P, d, 1.0)
        print(f" {d:4.2f} |  {r['ols_b_hat']:.4f}  |    {B['b_hat']:.4f}      |      {A['b_hat']:.4f}          ||  {r['ols_k_hat']:+.4f}  |  {B['phi_v']:+.4f}  |  {A['phi_v']:+.4f}")
    print("\nρ < 1 (AR(1) latent), d = 1 — naive one-step bias, finite-T exact")
    print(" ρ    | fitted b̂ | exact   || fitted φ_v | exact")
    for r in R["panelB"]:
        rho = r["rho"]; B = finite_T_moments(P, 1.0, rho)
        print(f" {rho:4.2f} |  {r['ols_b_hat']:.4f}  | {B['b_hat']:.4f} ||  {r['ols_k_hat']:+.4f}  | {B['phi_v']:+.4f}")
