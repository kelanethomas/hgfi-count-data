#!/usr/bin/env python3
"""Figure 1: fiducial and flat-prior densities on the local scale
h = sqrt(n) (theta - theta_0), for one simulated borderline dataset at each of
n = 20, 100 and 500 (rows) and the intercept, the continuous-covariate
coefficient and nu (columns), with a Normal density matched to the fiducial
draws. Observations are assigned by the mean rule, as in Sections 5.2-5.3.

The draws are saved to --data, so --replot redraws the figure without refitting.

    python sim_bvm_figure.py            # fit, save the draws and plot
    python sim_bvm_figure.py --replot   # plot from the saved draws
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
STAN_FILE      = os.path.join(THIS_DIR, "model_single_bin_hybrid.stan")
STAN_FILE_FLAT = os.path.join(THIS_DIR, "model_single_bin_hybrid_flat.stan")
sys.path.insert(0, THIS_DIR)
import sim_coverage as simcov  # noqa: E402

COVARIATE_NAMES = simcov.COVARIATE_NAMES

PARAM_LABELS = {
    "Intercept": "Intercept",
    "X_cont": "Continuous-Covariate Coefficient",
    "X_grpA": "Group A Coefficient",
    "X_grpB": "Group B Coefficient",
    "nu": r"Dispersion Parameter $\nu$",
}
DEFAULT_NU = simcov.DEFAULT_NU


def fit_dataset(exe, X, t, Y, threshold, chains, warmup, sampling, adapt_delta, seed,
                label=""):
    """Fit one model on a fixed dataset; return dict {param: draws}. Prints
    convergence diagnostics so the illustrative fits can be verified."""
    import cmdstanpy
    cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
    from cmdstanpy import CmdStanModel
    Y = Y.astype(int)
    # the mean rule (pooled rate x exposure), as in Sections 5.2-5.3
    rate_hat = float(np.mean(Y / t))
    norm_mask = (rate_hat * t) >= threshold
    nb_mask = ~norm_mask
    data = dict(
        r=X.shape[1],
        n_norm=int(norm_mask.sum()),
        X_norm=X[norm_mask].tolist(), t_norm=t[norm_mask].tolist(),
        Y_norm=Y[norm_mask].astype(float).tolist(),
        n_nb=int(nb_mask.sum()),
        X_nb=X[nb_mask].tolist(), t_nb=t[nb_mask].tolist(),
        Y_nb=Y[nb_mask].astype(int).tolist(),
    )
    model = CmdStanModel(exe_file=exe)
    # retry with a fresh seed if the representative fit does not converge (<= 1.05)
    for attempt in range(4):
        fit = model.sample(
            data=data, chains=chains, parallel_chains=chains,
            iter_warmup=warmup, iter_sampling=sampling,
            adapt_delta=adapt_delta, max_treedepth=12, seed=seed + 1000 * attempt,
            show_progress=False, show_console=False)
        if float(fit.summary()["R_hat"].max()) <= 1.05:
            break
    # report diagnostics for this illustrative fit
    s = fit.summary()
    dd = fit.draws_pd()
    divs = int(dd["divergent__"].sum())
    mtd = int((dd["treedepth__"] >= 12).sum())
    print(f"      {label:10s} R-hat={float(s['R_hat'].max()):.4f}  "
          f"ESS_bulk={float(s['ESS_bulk'].min()):.0f}  "
          f"ESS_tail={float(s['ESS_tail'].min()):.0f}  div={divs}  maxtd={mtd}")
    beta = fit.stan_variable("beta")
    out = {COVARIATE_NAMES[k]: beta[:, k] for k in range(len(COVARIATE_NAMES))}
    out["nu"] = fit.stan_variable("nu")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="borderline",
                    choices=list(simcov.REGIME_TARGET_MU))
    # n=20 is deliberately below the coverage-study grid: the smallest sample makes
    # the pre-asymptotic departure from the Gaussian limit visible. These are the
    # values used for the published figure.
    ap.add_argument("--n_grid", nargs="+", type=int, default=[20, 100, 500])
    ap.add_argument("--params", nargs="+", default=["Intercept", "X_cont", "nu"])
    ap.add_argument("--nu", type=float, default=DEFAULT_NU,
                    help="true dispersion, matching sim_coverage.py's --nu")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--sampling", type=int, default=4000)
    ap.add_argument("--adapt_delta", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "figures",
                                                  "bvm_local_scale"),
                    help="path prefix; writes <out>.png "
                         "(default: figures/bvm_local_scale)")
    ap.add_argument("--data", default=os.path.join(THIS_DIR, "sim_bvm_figure_draws.npz"),
                    help="where the fitted draws are saved, and read from with --replot")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved draws instead of refitting")
    args = ap.parse_args()

    if args.replot:
        z = np.load(args.data, allow_pickle=True)
        args.n_grid = [int(v) for v in z["n_grid"]]
        args.params = [str(v) for v in z["params"]]
        args.regime = str(z["regime"])
        true_by_param = z["true_by_param"].item()
        draws = {(p, n): {"gfd": z[f"gfd__{p}__{n}"], "bayes": z[f"bayes__{p}__{n}"]}
                 for p in args.params for n in args.n_grid}
        print(f"[replot] draws from {args.data}")
        return plot(args, draws, true_by_param)

    from cmdstanpy import CmdStanModel
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    exe_gfd = CmdStanModel(stan_file=STAN_FILE).exe_file
    exe_bayes = CmdStanModel(stan_file=STAN_FILE_FLAT).exe_file

    pilot_rng = np.random.default_rng(args.seed)
    beta_true = simcov.true_beta_for_regime(args.regime, pilot_rng, args.nu)
    true_by_param = dict(zip(COVARIATE_NAMES, beta_true))
    true_by_param["nu"] = args.nu
    print(f"[regime={args.regime}] beta_true={np.round(beta_true,3)}, nu={args.nu}")

    rng = np.random.default_rng(args.seed + 1)
    # fit one representative dataset per n, both methods
    draws = {}   # (param, n) -> {'gfd':..., 'bayes':...}
    for n in args.n_grid:
        X, t = simcov.make_synthetic_design(n, rng)
        eta = X @ beta_true
        Y = rng.negative_binomial(np.exp(eta) * t, args.nu / (1 + args.nu))
        print(f"  fitting n={n} ...")
        g = fit_dataset(exe_gfd, X, t, Y, 500.0, args.chains, args.warmup,
                        args.sampling, args.adapt_delta, args.seed + n, "GFD")
        b = fit_dataset(exe_bayes, X, t, Y, 500.0, args.chains, args.warmup,
                        args.sampling, args.adapt_delta, args.seed + n, "Flat-Prior Bayes")
        for p in args.params:
            draws[(p, n)] = {"gfd": g[p], "bayes": b[p]}

    # save the draws, so the figure can be redrawn without refitting (--replot)
    np.savez_compressed(
        args.data, n_grid=np.array(args.n_grid), params=np.array(args.params),
        regime=args.regime, seed=args.seed, true_by_param=np.array(true_by_param, dtype=object),
        **{f"{k}__{p}__{n}": v[k] for (p, n), v in draws.items() for k in ("gfd", "bayes")})
    print(f"[saved] draws -> {args.data}")
    return plot(args, draws, true_by_param)


def plot(args, draws, true_by_param):

    # plot: rows = sample sizes, cols = parameters  (matches Borgert & Hannig Fig 4)
    nr, nc = len(args.n_grid), len(args.params)
    # sized for the paper, where it prints 4.9 in wide (0.68 of this width), so
    # the tick labels print at about 7 pt; at most four x-ticks per panel so the
    # labels do not run together
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"xtick.labelsize": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 11.5})
    fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 1.95 * nr), squeeze=False)
    for i, n in enumerate(args.n_grid):
        for j, p in enumerate(args.params):
            ax = axes[i][j]
            th0 = true_by_param[p]
            for key, color, lbl in [("gfd", "tab:red", "GFD"),
                                     ("bayes", "tab:blue", "Flat-Prior Bayes")]:
                h = np.sqrt(n) * (draws[(p, n)][key] - th0)     # local scale
                h = h[np.isfinite(h)]
                # clip extreme tails (flat-Bayes can be near-improper) for a readable KDE
                lo, hi = np.percentile(h, [0.5, 99.5])
                hk = h[(h >= lo) & (h <= hi)]
                if hk.std() > 0:
                    xs = np.linspace(hk.min(), hk.max(), 200)
                    ax.plot(xs, gaussian_kde(hk)(xs), color=color, lw=1.6, label=lbl)
            # Normal reference: the BvM limit N(I^{-1}Delta, I^{-1}) on the local
            # scale has a NONZERO random mean for a single dataset. We center it at
            # the empirical local-scale mean (approx I^{-1}Delta) and scale by the
            # empirical local-scale sd (approx sqrt(I^{-1})) of the GFD draws.
            hg = np.sqrt(n) * (draws[(p, n)]["gfd"] - th0)
            xs = np.linspace(np.percentile(hg, 0.5), np.percentile(hg, 99.5), 200)
            ax.plot(xs, norm.pdf(xs, hg.mean(), hg.std()), "k--", lw=1.0, label="Normal")
            ax.set_yticks([])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if i == 0:
                ax.set_title(PARAM_LABELS.get(p, p).replace("-Covariate ", "-Covariate\n"))   # panel labels across the top
            if j == 0:
                ax.set_ylabel(f"$n = {n}$")  # sample size labels down the side

    fig.suptitle(f"Local-Scale Fiducial vs Flat-Prior Bayes Densities "
                 f"({args.regime.capitalize()} Regime), h = √n(θ−θ₀)", y=1.0, fontsize=12)
    fig.tight_layout()
    # one legend row under the grid: inside a panel it covers the n = 20 curves
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=10.5, frameon=False, handlelength=1.8, columnspacing=1.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    path = f"{args.out}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
