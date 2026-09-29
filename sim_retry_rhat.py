#!/usr/bin/env python3
"""Refit the Stan fits of a sweep that had not converged (R-hat above 1.05, or
no R-hat because a chain did not move), with more chains, a longer warmup, a
higher target acceptance rate and fresh chain seeds, and merge the refits back;
a fit is excluded only if it fails twice. Discrete-GFD rows have no R-hat and
are never refit.

Each dataset is regenerated from the sweep's seeding (a SeedSequence spawned per
(regime, n) cell in the order of --regimes) with nu and beta read from the
sweep's output, and checked against the sweep's record through every failed
rule's Normal-assigned count before any refit.

    python sim_retry_rhat.py --in sim_partition_shape_all.csv --n_jobs 48
    python sim_retry_rhat.py --in sim_coverage_full.csv --nu 0.05 --R 2000 --n_jobs 48
    python sim_retry_rhat.py --in sim_partition_shape_all_merged.csv --missing-only ...

--missing-only refits only the fits with no R-hat (a first retry selected
R-hat > 1.05 only; this is how those fits were refit afterwards).
Outputs <in>_retry.csv (the refits) and <in>_merged.csv (the sweep with each
converged refit replacing its first attempt).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from sim_coverage import (COVARIATE_NAMES, DEFAULT_N_GRID, DEFAULT_SLOPES,  # noqa: E402
                          REGIMES, STAN_FILE, STAN_FILE_NB, T_RANGE,
                          _norm_mask, make_synthetic_design, run_one,
                          true_beta_for_regime)

RHAT_MAX = 1.05


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="the sweep's results")
    ap.add_argument("--seed", type=int, default=20260616, help="the sweep's --seed (data seeds)")
    ap.add_argument("--R", type=int, default=1000, help="the sweep's --R")
    ap.add_argument("--regimes", nargs="+", default=REGIMES,
                    help="in the sweep's order (seed spawning depends on it)")
    ap.add_argument("--n_grid", nargs="+", type=int, default=DEFAULT_N_GRID)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=500.0)
    ap.add_argument("--nu", type=float, default=None,
                    help="the sweep's fixed nu, for a sweep that did not vary it (and so "
                         "recorded no nu_true); the true beta is then taken from the file")
    ap.add_argument("--sparse-shape", type=float, default=5.0, help="the sweep's --sparse-shape")
    ap.add_argument("--partitions", nargs="+", default=None,
                    help="refit only fits of these rules (default: all)")
    ap.add_argument("--missing-only", action="store_true",
                    help="refit only the fits with no R-hat")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.95)
    ap.add_argument("--stan_seed_offset", type=int, default=7919,
                    help="added to the data seed to seed the retry chains")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true",
                    help="verify the regenerated datasets and stop before fitting")
    a = ap.parse_args()
    slopes, t_range = DEFAULT_SLOPES, T_RANGE

    d = pd.read_csv(a.inp, low_memory=False)
    # a fit is one (dataset, partition, method)
    key = ["regime", "n", "rep", "partition", "method"]
    fits = d[d.param != "__fit_error__"].drop_duplicates(key)
    stan_fit = fits.fit_model != "discrete_gfd" if "fit_model" in fits else True
    bad = fits[stan_fit & (fits.rhat_max.isna() if a.missing_only
                           else ~(fits.rhat_max <= RHAT_MAX))]
    if a.partitions:
        bad = bad[bad.partition.isin(a.partitions)]
    print(f"[in] {len(fits):,} fits; {len(bad)} not converged "
          f"({int(bad.rhat_max.isna().sum())} with no R-hat) "
          f"on {len(bad.drop_duplicates(['regime', 'n', 'rep']))} datasets")
    print(bad.groupby(["regime", "n"]).size().unstack(fill_value=0).to_string())

    ss = np.random.SeedSequence(a.seed)
    cells = {(rg, n): ss.spawn(a.R) for rg in a.regimes for n in a.n_grid}

    # ---- regenerate each failed dataset and verify it against the record --
    jobs, discriminating = [], 0
    for (regime, n, rep), g in bad.groupby(["regime", "n", "rep"]):
        seed = int(cells[(regime, n)][rep].generate_state(1)[0])
        nu = a.nu if a.nu is not None else float(g.nu_true.iloc[0])
        rows = d[(d.regime == regime) & (d.n == n) & (d.rep == rep)
                 & (d.partition == g.partition.iloc[0])]
        beta = np.array([float(rows[rows.param == p]["true"].iloc[0]) for p in COVARIATE_NAMES])
        if a.nu is None:
            # varying nu: each replication's beta is recalibrated to its nu
            beta_chk = true_beta_for_regime(regime, np.random.default_rng(1000 + rep), nu, slopes,
                                            t_range, sparse_shape=a.sparse_shape)
            if not np.allclose(beta, beta_chk):
                sys.exit(f"true beta at {regime} n={n} rep={rep} is not the regime calibration "
                         f"for its nu")
        rng = np.random.default_rng(seed)
        X, t = make_synthetic_design(n, rng, t_range)
        alpha_t = np.exp(X @ beta) * t
        Y = rng.negative_binomial(alpha_t, nu / (1.0 + nu))
        for _, row in g.drop_duplicates("partition").iterrows():
            nm = _norm_mask(row.partition, Y, alpha_t, X, t, a.threshold, nu, a.kappa, X.shape[1] + 1)
            if int(nm.sum()) != int(row.n_norm):
                sys.exit(f"assignment mismatch at {regime} n={n} rep={rep} {row.partition}: "
                         f"regenerated n_norm={int(nm.sum())}, recorded {int(row.n_norm)}; "
                         f"--seed, --R, --regimes, --threshold or --kappa differ from the sweep")
            # a check that could have failed: some but not all observations assigned
            discriminating += int(0 < int(nm.sum()) < n)
        jobs.append(dict(rep=int(rep), regime=regime, n=int(n), seed=seed, nu=nu, beta=beta,
                         partitions=sorted(set(g.partition)), methods=sorted(set(g.method)),
                         rhat_first={(r.partition, r.method): float(r.rhat_max)
                                     for _, r in g.iterrows()}))
    print(f"[verify] all {len(jobs)} datasets regenerate to the recorded Normal-assigned counts "
          f"({len(bad)} fits); {discriminating} of the checks could have failed")
    # a rule that assigns everything or nothing matches whatever the seed; all
    # cells share one SeedSequence, so one discriminating match confirms them all
    if discriminating == 0:
        sys.exit("no assignment check could distinguish the seeds; the datasets are unverified")
    if a.dry_run:
        return

    # ---- refit ----------------------------------------------------------
    from cmdstanpy import CmdStanModel
    from joblib import Parallel, delayed
    exe_gfd = CmdStanModel(stan_file=STAN_FILE).exe_file
    exe_nb = CmdStanModel(stan_file=STAN_FILE_NB).exe_file
    print(f"[retry] chains={a.chains}, warmup={a.warmup}, sampling={a.sampling}, "
          f"adapt_delta={a.adapt_delta}, stan seed = data seed + {a.stan_seed_offset}")
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(run_one)(j["rep"], j["regime"], j["n"], exe_gfd, exe_gfd, exe_nb, j["beta"],
                         a.threshold, a.chains, a.warmup, a.sampling, a.adapt_delta,
                         j["seed"], j["partitions"], j["methods"], j["nu"], t_range, a.kappa,
                         stan_seed=j["seed"] + a.stan_seed_offset)
        for j in jobs)
    rows = [r for sub in res for r in sub]
    r = pd.DataFrame(rows)
    r["attempt"] = 2
    first = {(j["regime"], j["n"], j["rep"]) + pm: v for j in jobs for pm, v in j["rhat_first"].items()}
    r["rhat_first"] = [first.get((x.regime, x.n, x.rep, x.partition, x.method), np.nan)
                       for x in r.itertuples()]
    # keep only the refits of the fits that failed: a dataset refit for one
    # method also returns the other, whose first attempt had converged
    failed = set(map(tuple, bad[key].values))
    keep = np.array([tuple(v) in failed for v in r[key].values]) | (r.param == "__fit_error__").values
    r = r[keep]
    stem = a.inp[:-4] if a.inp.endswith(".csv") else a.inp
    r.to_csv(stem + "_retry.csv", index=False)
    print(f"[retry] {len(jobs)} datasets refit in {(time.time() - t0) / 60:.1f} min -> {stem}_retry.csv")

    # ---- merge ----------------------------------------------------------
    err = r[r.param == "__fit_error__"] if "param" in r else r.iloc[0:0]
    ok_rows = r[(r.param != "__fit_error__") & (r.rhat_max <= RHAT_MAX)]
    conv = ok_rows.drop_duplicates(key)[key]
    still = r[(r.param != "__fit_error__") & (r.rhat_max > RHAT_MAX)].drop_duplicates(key)
    print(f"[merge] converged on retry: {len(conv)} of {len(bad)}; still above {RHAT_MAX}: {len(still)}; "
          f"fit errors: {len(err)}")
    if len(still):
        print(still.groupby(["regime", "n"]).size().unstack(fill_value=0).to_string())
        print("  R-hat after retry:", np.round(np.sort(still.rhat_max.values), 3).tolist())
    m = d.merge(conv.assign(_replace=True), on=key, how="left")
    m = m[m._replace.isna()].drop(columns="_replace")
    cols = [c for c in d.columns]
    merged = pd.concat([m, ok_rows.reindex(columns=cols)], ignore_index=True)
    merged.to_csv(stem + "_merged.csv", index=False)
    fits_m = merged.drop_duplicates(key)
    print(f"[merge] {len(fits_m):,} fits in {stem}_merged.csv; "
          f"{int((fits_m.rhat_max > RHAT_MAX).sum())} still flagged "
          f"({100 * (fits_m.rhat_max > RHAT_MAX).mean():.2f}%)")
    print(f"\n[summary] {len(bad)} of {len(fits):,} fits had R-hat > {RHAT_MAX}; refit with "
          f"{a.chains} chains, {a.warmup} warmup iterations and target acceptance "
          f"{a.adapt_delta}, {len(conv)} converged and {len(still) + len(err)} remain excluded.")


if __name__ == "__main__":
    main()
