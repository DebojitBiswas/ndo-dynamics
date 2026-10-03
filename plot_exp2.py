#!/usr/bin/env python3
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "exp2_aggregate.json")))
worlds = ['AR1', 'PWC', 'SINE']; labels = {'AR1': 'AR(1) latent\n(well specified)', 'PWC': 'piecewise-constant\n(misspecified)', 'SINE': 'sinusoidal\n(misspecified)'}
ests = ['NAIVE', 'NDO-Z', 'DKF', 'DKF-B']
C = {'NAIVE': '#eb6834', 'NDO-Z': '#2a78d6', 'DKF': '#eda100', 'DKF-B': '#1baf7a'}
L = {'NAIVE': 'NAIVE (one-step simulator)', 'NDO-Z': 'NDO-Z (trajectory latent)', 'DKF': 'DKF (latent process, b learned)', 'DKF-B': 'DKF-B (latent process, b known)'}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
for ax, key, title in [(axes[0], 'rmse', "(a) Query inside the training support, ū ∈ [3, 6]"), (axes[1], 'ood', "(b) Query outside the training support, ū ∈ [7, 9]")]:
    x = np.arange(len(worlds)); w = 0.2
    for i, e in enumerate(ests):
        m = [A[k][e][f'{key}_mean'] for k in worlds]; s = [A[k][e][f'{key}_sd'] for k in worlds]
        bars = ax.bar(x + (i - 1.5)*w, m, w*0.9, color=C[e], label=L[e], yerr=s, error_kw=dict(ecolor="#52514e", lw=1, capsize=2))
        for b_, v_ in zip(bars, m):
            ax.text(b_.get_x() + b_.get_width()/2, v_*1.35, f"{v_:.2f}", ha='center', va='bottom', fontsize=7, color="#52514e")
    ax.set_yscale('log'); ax.set_ylim(0.2, 300); ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in worlds])
    ax.set_ylabel("interventional RMSE (m/s), log scale"); ax.grid(axis='x', visible=False)
    ax.set_title(title, loc='left', fontsize=9.5)
axes[0].legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.savefig(os.path.join(HERE, "exp2_figure.png"), dpi=200); print("saved")
