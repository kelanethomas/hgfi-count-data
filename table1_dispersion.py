"""Table 1 and the numbers of Section 5.3: coverage of the 95% fiducial intervals
under the mean rule (tau = 500) by regime, dispersion and n; the NB shape among
Normal-assigned observations; and, for the borderline regime at nu = 0.005, how
much too short the intervals are.

Coverage is averaged over the five parameters, excluding Stan fits not
converged after the refit (R-hat above 1.05, or none); discrete-GFD fits have no
R-hat and are kept. The shapes exp(X_i' beta) t_i are not stored in the sweeps,
so each dataset is regenerated from the sweep's seeding and checked against its
recorded Normal-assigned count. Shortness is 1 - (mean interval width) /
(2 x 1.96 x sd of the point estimates across replications), for the parameter
with the lowest coverage.

    python table1_dispersion.py \\
        0.005:sim_coverage_nu0.005_merged_filled.csv:1000 \\
        0.02:sim_coverage_nu0.02_merged_filled.csv:1000 \\
        0.05:sim_coverage_main_discrete_final.csv:2000
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_coverage import (COVARIATE_NAMES, DEFAULT_N_GRID, T_RANGE, _norm_mask,  # noqa: E402
                          make_synthetic_design)

REGIMES = ["heavy", "borderline"]


def load(path):
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.method == "gfd") & (d.param != "__fit_error__") & d.regime.isin(REGIMES)]
    first = d.groupby(["regime", "n", "rep"]).rhat_max.transform("first")
    stan = d.fit_model != "discrete_gfd"
    return d[~(stan & ~(first <= 1.05))], int((stan & ~(first <= 1.05)).groupby(
        [d.regime, d.n, d.rep]).first().sum())


def shapes(d, nu, R, seed, spawn_regimes, threshold):
    """Shapes of the Normal-assigned observations, per regime, over all n and reps."""
    ss = np.random.SeedSequence(seed)
    cells = {(rg, n): ss.spawn(R) for rg in spawn_regimes for n in DEFAULT_N_GRID}
    out = {rg: [] for rg in REGIMES}
    for (rg, n), g in d.groupby(["regime", "n"]):
        beta = np.array([float(g[g.param == p]["true"].iloc[0]) for p in COVARIATE_NAMES])
        for rep, n_norm in g.drop_duplicates("rep")[["rep", "n_norm"]].itertuples(index=False):
            rng = np.random.default_rng(int(cells[(rg, n)][rep].generate_state(1)[0]))
            X, t = make_synthetic_design(n, rng, T_RANGE)
            alpha_t = np.exp(X @ beta) * t
            Y = rng.negative_binomial(alpha_t, nu / (1.0 + nu))
            m = _norm_mask("plugin_rate", Y, alpha_t, X, t, threshold, nu, 10.0, X.shape[1] + 1)
            if int(m.sum()) != int(n_norm):
                sys.exit(f"nu={nu} {rg} n={n} rep={rep}: regenerated {int(m.sum())} "
                         f"Normal-assigned, recorded {int(n_norm)}")
            out[rg].append(alpha_t[m])
    return {rg: np.concatenate(v) for rg, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="+", help="nu:file:R")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--spawn-regimes", nargs="+", default=["heavy", "borderline", "sparse"])
    ap.add_argument("--threshold", type=float, default=500.0)
    a = ap.parse_args()

    rows, shp, lowest = [], {}, {}
    for spec in a.sweeps:
        nu, path, R = spec.split(":")
        nu, R = float(nu), int(R)
        d, excluded = load(path)
        s = shapes(d, nu, R, a.seed, a.spawn_regimes, a.threshold)
        cov = d.groupby(["regime", "n"]).cov95.mean().unstack()
        per = d.groupby(["regime", "param", "n"]).cov95.mean()
        for rg in REGIMES:
            shp[(rg, nu)] = s[rg]
            lowest[(rg, nu)] = per.loc[rg].min()
            rows.append((rg, nu, np.median(s[rg]), np.percentile(s[rg], 5),
                         *cov.loc[rg, DEFAULT_N_GRID].to_numpy()))
        print(f"nu={nu}: {path}, {excluded} non-converged Stan fits excluded")
        if nu == min(float(x.split(":")[0]) for x in a.sweeps):
            b = d[d.regime == "borderline"]
            worst = b.groupby("param").cov95.mean().idxmin()
            print(f"  borderline, most affected parameter ({worst}):")
            for n, g in b[b.param == worst].groupby("n"):
                short = 1 - g.width95.mean() / (2 * 1.96 * g.point.std(ddof=1))
                miss_lo = (g.u_rank > 0.975).mean() if g.u_rank.notna().any() else np.nan
                miss_hi = (g.u_rank < 0.025).mean() if g.u_rank.notna().any() else np.nan
                print(f"    n={n}: coverage {g.cov95.mean():.3f}, {100 * short:.1f}% too short; "
                      f"truth below / above the interval {miss_lo:.3f} / {miss_hi:.3f}")

    print(f"\n{'Regime':11} {'nu':>6} {'shape median (5th pct)':>24}" +
          "".join(f"{n:>8}" for n in DEFAULT_N_GRID))
    for rg, nu, med, p5, *c in sorted(rows, key=lambda r: (REGIMES.index(r[0]), r[1])):
        print(f"{rg:11} {nu:>6g} {med:>13.1f} ({p5:>5.1f})    " + "".join(f"{np.floor(v * 1000 + 0.5 + 1e-9) / 1000:>8.3f}" for v in c))

    good = [k for k in shp if np.median(shp[k]) >= 10]
    s_good = np.concatenate([shp[k] for k in good])
    print(f"\ncells with median shape >= 10: shapes below 7 {100 * np.mean(s_good < 7):.3f}%, "
          f"smallest {s_good.min():.2f}")
    for k in sorted(shp, key=lambda k: (k[1], REGIMES.index(k[0]))):
        print(f"  {k[0]:10} nu={k[1]:<6g} median shape {np.median(shp[k]):7.2f}, 5th pct "
              f"{np.percentile(shp[k], 5):7.2f}, smallest {shp[k].min():6.2f}; lowest "
              f"single-parameter coverage {lowest[k]:.3f}")

if __name__ == "__main__":
    main()
