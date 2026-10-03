#!/usr/bin/env python3
"""Fisher information for b and k as a function of the excitation level sigma_n (Sec. 5.3 text): where does the information about b come from?"""
import json, os, torch, exp1_confounded_vehicle as E, theory_fisher_curve as F
HERE = os.path.dirname(os.path.abspath(__file__)); res = []
print("sig_n  rho   se_b   se_k_free  se_k_known")
for sn in [0.1, 0.5, 1.0]:
    E.P['sig_n'] = sn; F.TRUE['sig_n'] = sn
    for rho in [0.0, 0.5, 0.8, 0.95]:
        f = F.info_at_truth(rho, N=2000); fk = F.info_at_truth(rho, N=2000, b_known=1.0)
        res.append(dict(sig_n=sn, rho=rho, se_b=f['b']['se1'], se_k_free=f['k']['se1'], se_k_known=fk['k']['se1']))
        print(f"{sn:4.1f}  {rho:4.2f}  {f['b']['se1']:.4f}  {f['k']['se1']:.4f}  {fk['k']['se1']:.4f}")
json.dump(res, open(os.path.join(HERE, "theory_fisher_sign.json"), "w"), indent=1)
