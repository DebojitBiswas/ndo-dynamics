#!/usr/bin/env python3
"""
Exact-arithmetic proof of generic local identifiability (Conjecture, clause 1, local version).

The map θ ↦ (E y, Γ(0), Γ(1), Γ(2)) from the 9 parameters θ = (b,k,g,kp,v*,d,ρ,σ_n,σ_w) to the first
moments of the stationary law of y_t = (v_t, u_t) is a rational function of θ.  Its Jacobian is a matrix
of rational functions; the set of θ at which its rank drops below the generic rank is a proper algebraic
subset (the zero set of all 9×9 minors), hence of measure zero.  Therefore: if the Jacobian has rank 9 at
ONE point with rational coordinates, computed in exact arithmetic, then it has rank 9 at generic θ, and the
model is locally identifiable almost everywhere in parameter space (Rothenberg 1971 / Glover–Willems 1974).

We also verify the exact rank drop at ρ = 0 (rank 8) and exhibit its null direction, and that fixing b
does not change the rank conclusion for the remaining 8 parameters.
"""
import sympy as sp

b, k, g, kp, vs, d, rho, sn, sw = sp.symbols('b k g k_p v_star d rho sigma_n sigma_w', real=True)
DT = sp.Rational(1, 10)
theta = [b, k, g, kp, vs, d, rho, sn, sw]

a = 1 - DT*(k + b*kp)
A = sp.Matrix([[a, DT*(b*d + g), DT*b], [0, rho, 0], [0, 0, 0]])
kappa = sp.Matrix([DT*kp*b*vs, 0, 0])
Q = sp.diag(sw**2, 1 - rho**2, sn**2)
H = sp.Matrix([[1, 0, 0], [-kp, d, 1]]); h0 = sp.Matrix([0, kp*vs])

# stationary covariance: solve S = A S A^T + Q  (6 unknowns, symmetric)
s11, s12, s13, s22, s23, s33 = sp.symbols('s11 s12 s13 s22 s23 s33')
S = sp.Matrix([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])
eqs = (S - A*S*A.T - Q)
sol = sp.solve([eqs[0, 0], eqs[0, 1], eqs[0, 2], eqs[1, 1], eqs[1, 2], eqs[2, 2]], [s11, s12, s13, s22, s23, s33], dict=True)[0]
S = S.subs(sol)

m = (sp.eye(3) - A).inv()*kappa
moments = list(H*m + h0)
G0 = H*S*H.T; moments += [G0[0, 0], G0[0, 1], G0[1, 1]]
Al = sp.eye(3)
for ell in range(1, 3):
    Al = Al*A; G = H*Al*S*H.T; moments += list(G)
F = sp.Matrix(moments)                      # 2 + 3 + 4 + 4 = 13 moment functions
J = F.jacobian(theta)                       # 13 x 9, rational in theta

def rank_at(point, cols=None):
    Jp = J.subs(point)
    if cols is not None: Jp = Jp[:, cols]
    Jp = sp.Matrix(Jp).applyfunc(sp.nsimplify)
    return Jp.rank(), Jp

pt = {b: sp.Rational(1), k: sp.Rational(3, 10), g: sp.Rational(1), kp: sp.Rational(4, 5), vs: sp.Rational(20),
      d: sp.Rational(1), rho: sp.Rational(9, 10), sn: sp.Rational(1, 2), sw: sp.Rational(1, 20)}
pt2 = {b: sp.Rational(7, 5), k: sp.Rational(1, 7), g: sp.Rational(3, 2), kp: sp.Rational(2, 3), vs: sp.Rational(17),
       d: sp.Rational(-2, 3), rho: sp.Rational(1, 3), sn: sp.Rational(2, 3), sw: sp.Rational(1, 9)}

r1, _ = rank_at(pt); r2, _ = rank_at(pt2)
print("rank of Jacobian (13 x 9) at rational point 1 (rho=9/10):", r1)
print("rank of Jacobian (13 x 9) at rational point 2 (rho=1/3, generic-looking):", r2)
pt0 = dict(pt); pt0[rho] = sp.Rational(0)
r0, J0 = rank_at(pt0)
print("rank at rho = 0:", r0)
ns = J0.nullspace()
if ns:
    v = ns[0]; v = v/ max(abs(x) for x in v)
    print("null direction at rho=0 (b,k,g,kp,v*,d,rho,sn,sw):", [sp.nsimplify(x) for x in v])
cols_no_b = [i for i, t in enumerate(theta) if t != b]
rb, _ = rank_at(pt, cols_no_b); rb0, _ = rank_at(pt0, cols_no_b)
print("rank with b fixed (8 params): rho=9/10 ->", rb, " ; rho=0 ->", rb0)
