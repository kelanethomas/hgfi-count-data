#!/usr/bin/env python3
"""Fold the sparse-regime discrete-GFD results into the coverage-sweep schema
of sim_coverage.py, which the table and figure scripts read.

sim_discrete_coverage.py writes one row per (replication, parameter) with the
discrete GFD and its flat-prior comparator side by side; this writes them as
`gfd` and `bayes_flat` rows, carrying log nu to the nu scale exactly through
the interval endpoints (quantiles are equivariant). Every discrete fit passes
downstream R-hat filters (rhat_max = 1), since all are included.

    python discrete_to_sweep.py \
        --discrete sparse_shape5_n50_final.csv sparse_shape5_n100_final.csv \
                   sparse_shape5_n200_final.csv sparse_shape5_n500_final.csv \
        --hb sim_coverage_full_merged.csv --out sim_coverage_main_discrete_final.csv

--hb appends the heavy and borderline rows of the Stan sweep, giving the file
plot_interval_lengths.py reads (Figure 3).
"""
import argparse
import argparse
import os

import numpy as np
import pandas as pd

# The sampler's parameter names and the sweep's.
PARAM_MAP = {"beta0": "Intercept", "beta1": "X_cont", "beta2": "X_grpA", "beta3": "X_grpB",
             "log_nu": "nu"}


def nu_scale(d, lo="gfd_lo", hi="gfd_hi", centre="gfd_median"):
    """(true, width95, point) with the log-nu rows carried to the nu scale; the
    interval is exactly [exp(lo), exp(hi)]. Coverage needs no transform."""
    is_nu = (d.param == "log_nu").to_numpy()
    true = d.truth.to_numpy(float).copy()
    l, h = d[lo].to_numpy(float), d[hi].to_numpy(float)
    width = h - l
    point = d[centre].to_numpy(float).copy()
    true[is_nu] = np.exp(true[is_nu])
    width[is_nu] = np.exp(h[is_nu]) - np.exp(l[is_nu])
    point[is_nu] = np.exp(point[is_nu])
    return true, width, point


def convert(path, method):
    """gfd or bayes_flat rows of one sim_discrete_coverage.py results file."""
    d = pd.read_csv(path)
    if "refused" in d:
        d = d[d.refused.isna()]
    pre = "gfd" if method == "gfd" else "flat"
    centre = "gfd_median" if method == "gfd" else "flat_mean"
    true, width, point = nu_scale(d, f"{pre}_lo", f"{pre}_hi", centre)
    return pd.DataFrame({
        "regime": "sparse", "n": d.n.to_numpy(), "rep": d.rep.to_numpy(), "method": method,
        "partition": "expected_shape", "nu_true": 0.05,
        "param": d.param.map(PARAM_MAP).to_numpy(), "true": true,
        "cov95": d[f"{pre}_cov"].to_numpy(), "cov90": np.nan, "width95": width,
        "point": point, "u_rank": np.nan, "n_norm": 0, "n_nb": d.n.to_numpy(),
        "rhat_max": 1.0,
        "fit_model": "discrete_gfd" if method == "gfd" else "flat_independence"})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrete", nargs="+", required=True,
                    help="sim_discrete_coverage.py results, one file per n")
    ap.add_argument("--hb", default=None,
                    help="coverage sweep whose heavy and borderline rows are appended")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(a.discrete)
    gfd = pd.concat([convert(f, "gfd") for f in files], ignore_index=True)
    out = pd.concat([gfd] + [convert(f, "bayes_flat") for f in files], ignore_index=True)
    if a.hb:
        h = pd.read_csv(a.hb, low_memory=False)
        h = h[h.regime.isin(["heavy", "borderline"])]
        out = pd.concat([h, out[out.columns.intersection(h.columns)]], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} rows -> {a.out}\n")
    print("discrete GFD coverage by parameter and n:")
    print(gfd.pivot_table(index="param", columns="n", values="cov95").round(3))


if __name__ == "__main__":
    main()
