#!/usr/bin/env python3
"""
Lemma 1 for a vector state (the two-state world of Experiment 5).

    x = (p, v);  x_{t+1} = x_t + Δ (A x_t + B u_t + G c_t) + w_t,   A = [[0,1],[0,-k]], B = (0,b), G = (0,g)
    u_t = K x_t + k_p v* + d c_t + n_t,  K = (0, -k_p);  c_t AR(1) with persistence ρ.
Population least squares of (x_{t+1}-x_t)/Δ on X_t = (p_t, v_t, u_t) (pooled, centred) gives
    [Â  B̂] = [A  B] + G γᵀ,   γ = Σ_XX⁻¹ Σ_Xc ∈ R³.
So (i) the kinematic row (G_1 = 0) is unbiased, (ii) the speed row acquires a spurious coefficient g γ_p on position,
a coordinate that does not enter its physics at all — position is a proxy for the accumulated latent.
Exact finite-horizon moments by propagating (p, v, c) through the closed loop; checked against large-N OLS.
"""
import json, os, numpy as np
import exp1_confounded_vehicle as E

HERE = os.path.dirname(os.path.abspath(__file__)); P = E.P; DT = P['dt']; T = P['T']

def exact_gamma(rho, d, sig_c=1.0):
    b, k, g, kp, vs, sn, sw = P['b'], P['k'], P['g'], P['kp'], P['v_star'], P['sig_n'], P['sig_w']
    a = 1 - DT*(k + b*kp)
    # state s = (p, v, c):  s' = F s + kappa + eta
    F = np.array([[1, DT, 0], [0, a, DT*(b*d + g)], [0, 0, rho]]); kappa = np.array([0, DT*kp*b*vs, 0])
    Q = np.diag([0.0, DT**2*b**2*sn**2 + sw**2, (1 - rho**2)*sig_c**2])
    m = np.array([0.0, vs, 0.0]); S = np.diag([0.0, 1.0, sig_c**2])
    # pooled second moments of (p, v, u, c) over t < T;  u = -kp v + kp vs + d c + n
    Hu = np.array([0, -kp, d]); M2 = np.zeros((4, 4)); M1 = np.zeros(4)
    for t in range(T):
        L = np.vstack([np.eye(3), Hu])                         # (p,v,c,u) without constants/noise
        # y = (p, v, c, u) = L s + (0, 0, 0, kp v* + n)
        mu_y = L @ m + np.array([0, 0, 0, kp*vs]); cov_y = L @ S @ L.T; cov_y[3, 3] += sn**2
        M2 += cov_y + np.outer(mu_y, mu_y); M1 += mu_y
        m = F @ m + kappa; S = F @ S @ F.T + Q
    M2 /= T; M1 /= T; C = M2 - np.outer(M1, M1)               # pooled covariance of (p, v, c, u)
    idx_X = [0, 1, 3]; Sxx = C[np.ix_(idx_X, idx_X)]; Sxc = C[idx_X, 2]
    gamma = np.linalg.solve(Sxx, Sxc)
    Astar = np.array([[0, 1, 0], [0, -k, b]], dtype=float)    # true [A B] rows: p-row, v-row over (p, v, u)
    G = np.array([0, g]); return Astar + np.outer(G, gamma), gamma

def ols_fit(rho, d, N=20000, seed=0):
    import exp5_twostate as X5
    rng = np.random.default_rng(seed); c = np.zeros((N, T+1)); c[:, 0] = rng.normal(0, 1, N)
    for t in range(1, T+1): c[:, t] = rho*c[:, t-1] + np.sqrt(1 - rho**2)*rng.normal(0, 1, N)
    v = np.zeros((N, T+1)); p = np.zeros((N, T+1)); u = np.zeros((N, T)); v[:, 0] = P['v_star'] + rng.normal(0, 1, N)
    for t in range(T):
        u[:, t] = P['kp']*(P['v_star'] - v[:, t]) + d*c[:, t] + rng.normal(0, P['sig_n'], N)
        p[:, t+1] = p[:, t] + DT*v[:, t]
        v[:, t+1] = v[:, t] + DT*(P['b']*u[:, t] - P['k']*v[:, t] + P['g']*c[:, t]) + rng.normal(0, P['sig_w'], N)
    X = np.stack([p[:, :-1].ravel(), v[:, :-1].ravel(), u.ravel(), np.ones(N*T)], 1)
    Y = np.stack([((p[:, 1:] - p[:, :-1])/DT).ravel(), ((v[:, 1:] - v[:, :-1])/DT).ravel()], 1)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]; return beta[:3].T

if __name__ == "__main__":
    out = []
    print(" ρ    d  |   speed row: coef on p (true 0)   coef on v (true -0.3)   coef on u (true 1)  | kinematic row max |err| ")
    for rho, d in [(1.0, 1.0), (0.9, 1.0), (0.9, 0.5), (0.5, 1.0)]:
        ex, gam = exact_gamma(rho, d); fit = ols_fit(rho, d)
        print(f"{rho:4.2f} {d:4.2f} |  exact {ex[1,0]:+.3f} fit {fit[1,0]:+.3f}   exact {ex[1,1]:+.3f} fit {fit[1,1]:+.3f}   exact {ex[1,2]:+.3f} fit {fit[1,2]:+.3f}  |  {np.abs(fit[0]-np.array([0,1,0])).max():.4f}")
        out.append(dict(rho=rho, d=d, exact=ex.tolist(), fit=fit.tolist(), gamma=gam.tolist()))
    json.dump(out, open(os.path.join(HERE, "theory_lemma1_vector.json"), "w"), indent=1)
