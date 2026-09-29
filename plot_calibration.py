#!/usr/bin/env python3
"""Figure 2: empirical against nominal coverage of the fiducial and flat-prior
intervals in the borderline regime, at n = 50 and 500 (rows) for the intercept,
the continuous-covariate coefficient and nu (columns), with a pointwise 95%
Monte Carlo band. A central interval at level L covers the truth when
|u_rank - 0.5| <= L/2, u_rank being the fiducial CDF at the truth.

    python plot_calibration.py      # writes figures/calibration_borderline.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Manuscript-facing panel labels; keys are the raw parameter names in the CSV.
PARAM_LABELS = {
    "Intercept": "Intercept",
    "X_cont": "Continuous-Covariate Coefficient",
    "X_grpA": "Group A Coefficient",
    "X_grpB": "Group B Coefficient",
    "nu": r"Dispersion Parameter $\nu$",
}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(THIS_DIR, "sim_coverage_main_discrete_final.csv"),
                    help="coverage sweep (discrete_to_sweep.py output)")
    ap.add_argument("--regime", default="borderline")
    ap.add_argument("--n_grid", nargs="+", type=int, default=[50, 500],
                    help="sample sizes to show (default: 50 and 500, as in the paper)")
    ap.add_argument("--params", nargs="+", default=["Intercept", "X_cont", "nu"])
    ap.add_argument("--levels", type=int, default=25,
                    help="number of nominal levels between 0.05 and 0.99")
    ap.add_argument("--rhat_max", type=float, default=1.05,
                    help="drop fits with rhat_max above this before plotting")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "figures",
                                                  "calibration"),
                    help="path prefix (default: figures/calibration)")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    d = d[d.get("param") != "__fit_error__"].copy()
    if "rhat_max" in d.columns:
        n0 = len(d)
        n_fits = d[d.rhat_max > args.rhat_max].drop_duplicates(
            [c for c in ["regime", "n", "rep", "method"] if c in d]).shape[0]
        d = d[d.rhat_max <= args.rhat_max]
        print(f"[filter] dropped {n0 - len(d)} rows ({n_fits} fits) with "
              f"rhat_max > {args.rhat_max}")
    if "u_rank" not in d.columns:
        raise SystemExit("CSV has no 'u_rank' column — re-run sim_coverage.py "
                         "(updated version) to record it.")
    if "method" not in d.columns:
        d["method"] = "gfd"

    d = d[d.regime == args.regime]
    n_grid = sorted(d.n.unique()) if args.n_grid is None else args.n_grid
    levels = np.linspace(0.05, 0.99, args.levels)

    def coverage_curve(sub):
        # empirical coverage at each nominal level L from the u_rank values
        u = sub["u_rank"].values
        return np.array([np.mean(np.abs(u - 0.5) <= L / 2) for L in levels])

    styles = {"gfd": ("tab:red", "GFD"), "bayes_flat": ("tab:blue", "Flat-Prior Bayes")}

    # rows are sample sizes and columns parameters, as in Figures 1 and 3; sized for
    # the paper, where it prints 4.9 in wide, so the tick labels print at about 7 pt
    nr, nc = len(n_grid), len(args.params)
    plt.rcParams.update({"xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "axes.labelsize": 11, "axes.titlesize": 11.5})
    fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 2.35 * nr), squeeze=False)
    for i, n in enumerate(n_grid):
        for j, p in enumerate(args.params):
            ax = axes[i][j]
            # Monte Carlo uncertainty band around the diagonal: at nominal level L,
            # coverage is a proportion of R replications, so MCSE = sqrt(L(1-L)/R).
            R = len(d[(d.param == p) & (d.n == n) & (d.method == "gfd")])
            if R > 0:
                mcse = np.sqrt(levels * (1 - levels) / R)
                ax.fill_between(levels,
                                np.maximum(0.0, levels - 1.96 * mcse),
                                np.minimum(1.0, levels + 1.96 * mcse),
                                color="gray", alpha=0.20, lw=0,
                                label="95% MC Band")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Exact")
            for method, (color, lbl) in styles.items():
                sub = d[(d.param == p) & (d.n == n) & (d.method == method)]
                if len(sub) == 0:
                    continue
                cov = coverage_curve(sub)
                ax.plot(levels, cov, color=color, lw=1.8, label=lbl)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
            if i == 0:
                ax.set_title(PARAM_LABELS.get(p, p).replace("-Covariate ", "-Covariate\n"))
            if i == nr - 1:
                ax.set_xlabel("Nominal Level")
            if j == 0:
                ax.set_ylabel(f"$n = {n}$\nEmpirical Coverage")
    fig.suptitle(f"Coverage Calibration: GFD vs Flat-Prior Bayes ({args.regime.capitalize()} Regime)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               fontsize=10.5, frameon=False, handlelength=1.8, columnspacing=1.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    path = f"{args.out}_{args.regime}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[saved] {path}")

    # also print the 95% slice as a quick table
    print(f"\n95% coverage by parameter x n x method (regime={args.regime}):")
    tbl = (d[d.param.isin(args.params)]
           .groupby(["param", "n", "method"])["cov95"].mean().round(3)
           .unstack("method"))
    print(tbl.to_string())


if __name__ == "__main__":
    main()
