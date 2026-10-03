#!/usr/bin/env python3
import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); R=json.load(open(os.path.join(HERE,"theory_fisher_curve.json")))
rho=[r['rho'] for r in R]
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#b5b4ae","xtick.color":"#52514e","ytick.color":"#52514e","axes.grid":True,"grid.color":"#e6e5e0","grid.linewidth":0.6})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),constrained_layout=True)
ax[0].plot(rho,[r['se_b'] for r in R],color="#2a78d6",lw=2,marker="o",ms=3.5,label="input gain b (b free)")
ax[0].set_xlabel("latent persistence ρ"); ax[0].set_ylabel("asymptotic s.e. per trajectory"); ax[0].set_title("(a) Input gain: interior maximum of the s.e.",loc='left',fontsize=9.5)
i=int(np.argmax([r['se_b'] for r in R])); ax[0].annotate(f"max at ρ = {rho[i]}",(rho[i],R[i]['se_b']),xytext=(0.25,R[i]['se_b']*0.98),fontsize=8,color="#52514e",arrowprops=dict(arrowstyle='-',color="#b5b4ae"))
ax[0].legend(frameon=False,fontsize=7.5)
ax[1].plot(rho,[r['se_k_free'] for r in R],color="#eda100",lw=2,marker="o",ms=3.5,label="drag k, b learned")
ax[1].plot(rho,[r['se_k_known'] for r in R],color="#1baf7a",lw=2,marker="o",ms=3.5,label="drag k, b known (P1)")
ax[1].set_xlabel("latent persistence ρ"); ax[1].set_ylabel("asymptotic s.e. per trajectory"); ax[1].set_title("(b) Drag coefficient, with and without P1",loc='left',fontsize=9.5); ax[1].legend(frameon=False,fontsize=7.5)
fig.savefig(os.path.join(HERE,"fisher_figure.png"),dpi=200); print("saved")
