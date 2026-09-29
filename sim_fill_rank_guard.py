"""Discrete-GFD fits for the coverage-sweep datasets on which the partition rule
leaves fewer than K + 1 observations in the Normal component (Section 5.3).

Such a fit is entirely NB, and the paper computes it with the discrete
construction. In a sweep made with sim_coverage.py it appears either as the
NB-only Stan model (fit_model "discrete_nb", nothing assigned to the Normal
component) or, in sweeps that predate the rank guard, as a missing dataset
(1 to K assigned). This fits every such dataset with the discrete-GFD sampler
(sim_discrete_coverage.fit_dataset) and puts the fit in its place.

Each dataset is regenerated from the sweep's seeding (one SeedSequence, R
children per (regime, n) cell, regimes in the sweep's order) with the true beta
recorded in the file, and accepted only if the rule, before the guard, assigns
at most K observations to the Normal component. Rows get fit_model
"discrete_gfd"; discrete fits have no R-hat, and cov90 is not computed.

    python sim_fill_rank_guard.py --in sim_coverage_nu0.005_merged.csv --nu 0.005 \\
        --R 1000 --regimes heavy borderline --n_jobs 48

Writes <in>_filled.csv (the sweep with the discrete fits in place) and
<in>_rankguard.csv (the discrete fits).
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_coverage import (COVARIATE_NAMES, DEFAULT_N_GRID, T_RANGE, _raw_norm_mask,  # noqa: E402
                          make_synthetic_design)
from discrete_to_sweep import PARAM_MAP  # noqa: E402


def fit_one(job, steps):
    sys.path.insert(0, HERE)
    from sim_discrete_coverage import fit_dataset, dataset_base
    rng = np.random.default_rng(job["seed"])
    X, t = make_synthetic_design(job["n"], rng, T_RANGE)
    Y = rng.negative_binomial(np.exp(X @ job["beta"]) * t,
                              job["nu"] / (1.0 + job["nu"])).astype(float)
    truth = np.append(job["beta"], np.log(job["nu"]))
    rows = fit_dataset(X, Y, t, truth, dataset_base(job["rep"], X, Y), steps,
                       chain_seed=job["seed"] % (2 ** 31), rng_flat=None)
    out = []
    for r in rows:
        if r.get("refused"):
            return [dict(regime=job["regime"], n=job["n"], rep=job["rep"], method="gfd",
                         partition=job["partition"], param="__fit_error__",
                         error=f"discrete fit refused: {r['refused']}")]
        p = PARAM_MAP[r["param"]]
        lo, hi = r["gfd_lo"], r["gfd_hi"]
        if p == "nu":                                  # log-nu interval carried to nu
            lo, hi, point, true = np.exp(lo), np.exp(hi), np.exp(r["gfd_median"]), job["nu"]
        else:
            point, true = r["gfd_median"], r["truth"]
        out.append(dict(regime=job["regime"], n=job["n"], rep=job["rep"], method="gfd",
                        partition=job["partition"], nu_true=job["nu"], param=p, true=true,
                        cov95=int(lo <= true <= hi), cov90=np.nan, width95=hi - lo,
                        point=point, n_norm=0, n_nb=job["n"], rhat_max=np.nan,
                        fit_model="discrete_gfd"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--nu", type=float, required=True, help="the sweep's fixed nu")
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--regimes", nargs="+", default=["heavy", "borderline", "sparse"],
                    help="in the sweep's order (seed spawning depends on it)")
    ap.add_argument("--n_grid", nargs="+", type=int, default=DEFAULT_N_GRID)
    ap.add_argument("--threshold", type=float, default=500.0)
    ap.add_argument("--partition", default="plugin_rate")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    d = pd.read_csv(a.inp, low_memory=False)
    g = d[(d.method == "gfd") & (d.param != "__fit_error__")]
    ss = np.random.SeedSequence(a.seed)
    K = len(COVARIATE_NAMES)
    jobs = []
    for regime in a.regimes:
        for n in a.n_grid:
            kids = ss.spawn(a.R)
            cell = g[(g.regime == regime) & (g.n == n)]
            if cell.empty:
                continue
            nb_only = set(cell[cell.fit_model == "discrete_nb"].rep)
            targets = sorted((set(range(a.R)) - set(cell.rep)) | nb_only)
            beta = np.array([float(cell[cell.param == p]["true"].iloc[0]) for p in COVARIATE_NAMES])
            for rep in targets:
                seed = int(kids[rep].generate_state(1)[0])
                rng = np.random.default_rng(seed)
                X, t = make_synthetic_design(n, rng, T_RANGE)
                alpha_t = np.exp(X @ beta) * t
                Y = rng.negative_binomial(alpha_t, a.nu / (1.0 + a.nu))
                k = int(_raw_norm_mask(a.partition, Y, alpha_t, X, t, a.threshold, a.nu).sum())
                if k > K or (rep in nb_only and k != 0):
                    sys.exit(f"{regime} n={n} rep={rep}: the rule assigns {k} observations, which "
                             f"does not explain the recorded fit; the sweep is not reproduced")
                jobs.append(dict(regime=regime, n=n, rep=rep, seed=seed, beta=beta, nu=a.nu,
                                 partition=a.partition, k=k))
    if not jobs:
        print("[verify] no datasets to fit")
        return
    print(f"[verify] {len(jobs)} datasets with at most {K} observations assigned to Normal: "
          f"{sum(j['k'] == 0 for j in jobs)} with none (NB-only Stan fits replaced), "
          f"{sum(j['k'] > 0 for j in jobs)} with 1 to {K} (missing from the sweep)")
    if a.dry_run:
        return
    from joblib import Parallel, delayed
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs)(delayed(fit_one)(j, a.steps) for j in jobs)
    new = pd.DataFrame([r for rr in res for r in rr])
    replaced = pd.MultiIndex.from_tuples([(j["regime"], j["n"], j["rep"]) for j in jobs])
    idx = pd.MultiIndex.from_frame(d[["regime", "n", "rep"]])
    drop = (d.method == "gfd") & (d.partition == a.partition) & idx.isin(replaced)
    stem = a.inp[:-4] if a.inp.endswith(".csv") else a.inp
    new.to_csv(stem + "_rankguard.csv", index=False)
    pd.concat([d[~drop], new], ignore_index=True).to_csv(stem + "_filled.csv", index=False)
    err = int((new.param == "__fit_error__").sum())
    print(f"[fit] {len(jobs)} datasets in {(time.time() - t0) / 60:.1f} min; errors {err}; "
          f"-> {stem}_filled.csv")


if __name__ == "__main__":
    main()
