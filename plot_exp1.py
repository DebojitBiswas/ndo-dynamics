#!/usr/bin/env python3
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "exp1_results.json")))
u = np.array(R["u_grid"]); gt = np.array(R["gt"])
C = {"NAIVE": "#eb6834", "NDO-Z": "#2a78d6", "NDO-ZB": "#1baf7a"}
L = {"NAIVE": "Naive neural simulator", "NDO-Z": "NDO + trajectory latent", "NDO-ZB": "NDO + latent + known input gain"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

# (a) do-curves at d=1, rho=1
a = next(r for r in R["panelA"] if abs(r["d"] - 1.0) < 1e-9)
ax[0].plot(u, gt, color="#0b0b0b", lw=2, ls="--", label="Ground truth  E[v_T | do(u)]")
for k in C:
    ax[0].plot(u, a[k]["curve"], color=C[k], lw=2, marker="o", ms=4, label=L[k])
ax[0].set_xlabel("intervened input  ū  (m/s²)"); ax[0].set_ylabel("E[v_T | do(u = ū)]  (m/s)")
ax[0].set_title("(a) Interventional response, d = 1, constant latent", loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")

# (b) RMSE vs d
ds = [r["d"] for r in R["panelA"]]
for k in C:
    ax[1].plot(ds, [r[k]["rmse"] for r in R["panelA"]], color=C[k], lw=2, marker="o", ms=4, label=L[k])
ax[1].set_yscale("log"); ax[1].set_xlabel("confounding strength  d  (policy's use of latent c)")
ax[1].set_ylabel("do-curve RMSE (m/s), log scale")
ax[1].set_title("(b) Constant latent (ρ = 1): time buys identification", loc="left", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=7.5, loc="center right")

# (c) RMSE vs rho
rhos = [r["rho"] for r in R["panelB"]]
for k in C:
    ax[2].plot(rhos, [r[k]["rmse"] for r in R["panelB"]], color=C[k], lw=2, marker="o", ms=4, label=L[k])
ax[2].set_yscale("log"); ax[2].invert_xaxis()
ax[2].set_xlabel("latent persistence  ρ  (1 = constant, 0 = white)")
ax[2].set_ylabel("do-curve RMSE (m/s), log scale")
ax[2].set_title("(c) Time-varying latent, d = 1: known input gain restores most of it", loc="left", fontsize=9.5)
ax[2].legend(frameon=False, fontsize=7.5)

fig.savefig(os.path.join(HERE, "exp1_figure.png"), dpi=200)
print("saved")
