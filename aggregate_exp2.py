#!/usr/bin/env python3
"""Aggregate Experiment 2 over seeds -> exp2_aggregate.json (mean/sd of in-support and OOD RMSE, learned gains)."""
import json, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "exp2_results_seed*.json")))
if os.path.exists(os.path.join(HERE, "exp2_results.json")): files.append(os.path.join(HERE, "exp2_results.json"))
R = [json.load(open(f)) for f in files]
ests = ['NAIVE', 'NDO-Z', 'DKF', 'DKF-B']; agg = {}
for w in ['AR1', 'PWC', 'SINE']:
    agg[w] = {}
    for e in ests:
        r = [x[w][e]['rmse'] for x in R if w in x]; o = [x[w][e]['rmse_ood'] for x in R if w in x and 'rmse_ood' in x[w][e]]
        d = dict(rmse=r, rmse_mean=float(np.mean(r)), rmse_sd=float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                 ood=o, ood_mean=float(np.mean(o)) if o else float('nan'), ood_sd=float(np.std(o, ddof=1)) if len(o) > 1 else 0.0)
        if e.startswith('DKF'):
            d['dfdu'] = [x[w][e]['dfdu'] for x in R if w in x]; d['dfdu_mean'] = float(np.mean(d['dfdu']))
        agg[w][e] = d
json.dump(agg, open(os.path.join(HERE, "exp2_aggregate.json"), "w"), indent=1); print("wrote exp2_aggregate.json from", len(R), "seed files")
