#!/usr/bin/env python3
"""Figure 3: 95% interval lengths of the fiducial and flat-prior intervals by
regime (rows) and parameter (columns), from the sweep built by
discrete_to_sweep.py (the sparse rows are the discrete construction).

    python plot_interval_lengths.py   # writes figures/interval_lengths_byregime.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Manuscript-facing panel labels; keys are the raw parameter names in the CSV.
PARAM_LABELS = {
    "Intercept": "Intercept",
    "X_cont": "Continuous-Covariate\nCoefficient",   # two lines: the panels are narrow
    "X_grpA": "Group A Coefficient",
    "X_grpB": "Group B Coefficient",
    "nu": r"Dispersion Parameter $\nu$",
}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(THIS_DIR, "sim_coverage_main_discrete_final.csv"),
                    help="the sweep with the sparse rows from the discrete GFD "
                         "(built as described in discrete_to_sweep.py)")
    ap.add_argument("--regimes", nargs="+", default=["sparse", "borderline", "heavy"],
                    help="regimes to show, one row each (default: all three)")
    ap.add_argument("--params", nargs="+", default=["Intercept", "X_cont", "nu"])
    ap.add_argument("--n_grid", nargs="+", type=int, default=None,
                    help="sample sizes to show (default: all present)")
    ap.add_argument("--rhat_max", type=float, default=1.05,
                    help="drop fits with rhat_max above this before plotting")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "figures",
                                                  "interval_lengths"),
                    help="path prefix; the file is <out>_byregime.png (default: "
                         "figures/interval_lengths)")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    d = d[d.get("param") != "__fit_error__"]
    if "method" not in d.columns:
        raise SystemExit("CSV has no 'method' column — need a paired GFD/Bayes run.")
    if "rhat_max" in d.columns:
        n0 = len(d)
        n_fits = d[d.rhat_max > args.rhat_max].drop_duplicates(
            [c for c in ["regime", "n", "rep", "method"] if c in d]).shape[0]
        d = d[d.rhat_max <= args.rhat_max]
        print(f"[filter] dropped {n0 - len(d)} rows ({n_fits} fits) with "
              f"rhat_max > {args.rhat_max}")
    methods = [("gfd", "tab:red", "GFD"), ("bayes_flat", "tab:blue", "Flat-Prior Bayes")]

    nr, nc = len(args.regimes), len(args.params)
    # sized for the paper, where it prints 4.9 in wide (0.64 of this width), so
    # the tick labels print at about 7 pt and the titles at about 7.5 pt
    plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.labelsize": 11.5, "axes.titlesize": 12})
    fig, axes = plt.subplots(nr, nc, figsize=(2.55 * nc, 2.25 * nr), squeeze=False)
    for i, regime in enumerate(args.regimes):
        dr = d[d.regime == regime]
        ngrid = sorted(dr.n.unique()) if args.n_grid is None else args.n_grid
        for j, p in enumerate(args.params):
            ax = axes[i][j]
            for mi, (m, color, _) in enumerate(methods):
                data = [dr[(dr.param == p) & (dr.n == n) & (dr.method == m)]["width95"].values
                        for n in ngrid]
                pos = [k * 3 + mi for k in range(len(ngrid))]
                bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True,
                                showfliers=False, medianprops=dict(color="black"))
                for box in bp["boxes"]:
                    box.set(facecolor=color, alpha=0.5)
            ax.set_xticks([k * 3 + 0.5 for k in range(len(ngrid))])
            ax.set_xticklabels(ngrid)
            if i == 0:
                ax.set_title(PARAM_LABELS.get(p, p))   # parameter across the top
            if i == nr - 1:
                ax.set_xlabel("Sample Size $n$")
            if j == 0:
                ax.set_ylabel(f"{regime.capitalize()}\n95% Interval Width")  # regime down the side

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.5) for _, c, _ in methods]
    axes[0][-1].legend(handles, [l for _, _, l in methods], fontsize=10.5, frameon=False,
                       handlelength=1.2, borderaxespad=0.2)
    fig.suptitle("95% Interval Lengths: GFD vs Flat-Prior Bayes", y=1.0, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    suffix = args.regimes[0] if len(args.regimes) == 1 else "byregime"
    path = f"{args.out}_{suffix}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
