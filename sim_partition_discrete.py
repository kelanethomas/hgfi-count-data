#!/usr/bin/env python3
"""Discrete GFD for every entirely-NB fit of the partition-rule comparison (Table 2).

In the Stan sweep (sim_coverage.py --nu-range) a fit that assigns nothing to the
Normal component is the NB likelihood alone. This fits the discrete GFD to the
same datasets, once per dataset, recording it against every rule that assigned
nothing on it (merge_partition_discrete.py puts the fits into the sweep).

Datasets are regenerated from the sweep's seeding (a SeedSequence spawned per
(regime, n) cell in the sweep's order, so --spawn-regimes must list the regimes
in the order the sweep ran them), with nu and beta read from the sweep's output.
Before any fit, every rule's Normal-assigned count on the regenerated data is
checked against the sweep's record, and the script stops on any mismatch.

    # heavy + borderline arm (the sweep ran heavy, borderline, sparse)
    python sim_partition_discrete.py --inp sim_partition_shape_all_merged.csv \
        --spawn-regimes heavy borderline sparse --fit-regimes heavy borderline \
        --n 500 --rep-lo 0 --rep-hi 500 --out part_disc_hb_n500_a.csv

    # sparse arm (that sweep ran sparse alone, shape-calibrated)
    python sim_partition_discrete.py --inp sim_partition_sparse_shape5_merged.csv \
        --spawn-regimes sparse --fit-regimes sparse --sparse-shape 5 \
        --n 50 --out part_disc_sp_n50.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_coverage import (COVARIATE_NAMES, DEFAULT_N_GRID, DEFAULT_SLOPES,  # noqa: E402
                          T_RANGE, _norm_mask, make_synthetic_design,
                          true_beta_for_regime)
from sim_discrete_coverage import dataset_base, fit_dataset, check_resume  # noqa: E402
from discrete_settings import SIM_CAP, SIM_SCAN_SPACING  # noqa: E402

RULES = ["expected_shape", "plugin_rate", "plugin_shape", "plugin_shape_lcb"]


def regenerate(job, t_range):
    """Rebuild one dataset exactly as the sweep did."""
    rng = np.random.default_rng(job["seed"])
    X, t = make_synthetic_design(job["n"], rng, t_range)
    alpha_t = np.exp(X @ job["beta"]) * t
    Y = rng.negative_binomial(alpha_t, job["nu"] / (1.0 + job["nu"]))
    return X, t, alpha_t, Y


def fit_job(job, steps, chains, t_range, cap, scan_spacing):
    X, t, alpha_t, Y = regenerate(job, t_range)
    truth = np.append(job["beta"], np.log(job["nu"]))
    rows = fit_dataset(X, Y.astype(float), t, truth, dataset_base(job["rep"], X, Y),
                       steps, chain_seed=job["seed"] + 7919, chains=chains,
                       rng_flat=None, cap=cap,
                       scan_spacing=scan_spacing)
    for r in rows:
        r.update(regime=job["regime"], n=job["n"], nu_true=job["nu"],
                 rules=";".join(job["rules"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True, help="the sweep's (merged) partition CSV")
    ap.add_argument("--spawn-regimes", nargs="+", required=True,
                    help="regimes in the ORIGINAL sweep's order, for the seed spawn")
    ap.add_argument("--fit-regimes", nargs="+", required=True,
                    help="regimes to fit in this invocation")
    ap.add_argument("--n", type=int, nargs="+", default=None,
                    help="sample sizes to fit (default: all in the sweep's grid)")
    ap.add_argument("--n-grid", type=int, nargs="+", default=DEFAULT_N_GRID,
                    help="the sweep's full n grid, for the seed spawn")
    ap.add_argument("--rep-lo", type=int, default=0)
    ap.add_argument("--rep-hi", type=int, default=None, help="exclusive")
    ap.add_argument("--seed", type=int, default=20260616, help="the sweep's --seed")
    ap.add_argument("--R", type=int, default=1000, help="the sweep's --R")
    ap.add_argument("--sparse-shape", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=500.0)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--cap", nargs=2, type=float, default=list(SIM_CAP),
                    help="bounds on log nu (part of the parameter set Theta)")
    ap.add_argument("--scan_spacing", type=float, default=SIM_SCAN_SPACING,
                    help="log-nu spacing of the scan that locates the pieces of I(u)")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true",
                    help="regenerate and verify every dataset, then stop")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing results file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_range = T_RANGE
    n_fit = set(a.n) if a.n else set(a.n_grid)
    rep_hi = a.R if a.rep_hi is None else a.rep_hi

    d = pd.read_csv(a.inp, low_memory=False)
    d = d[d.method == "gfd"] if "method" in d else d
    fits = d.drop_duplicates(["regime", "n", "rep", "partition"])

    # ---- the sweep's data seeds, spawned in its original order -------------
    ss = np.random.SeedSequence(a.seed)
    cells = {(rg, n): ss.spawn(a.R) for rg in a.spawn_regimes for n in a.n_grid}

    # ---- datasets on which at least one rule assigned nothing to Normal ------
    sel = fits[fits.regime.isin(a.fit_regimes) & fits.n.isin(n_fit)
               & (fits.rep >= a.rep_lo) & (fits.rep < rep_hi)]
    jobs, t0 = [], time.time()
    for (rg, n, rep), g in sel.groupby(["regime", "n", "rep"]):
        rules = sorted(g[g.n_norm == 0].partition.tolist())
        if not rules:
            continue
        nu = float(g.nu_true.iloc[0])
        rec = d[(d.regime == rg) & (d.n == n) & (d.rep == rep)
                & (d.partition == g.partition.iloc[0])]
        beta = np.array([float(rec[rec.param == p]["true"].iloc[0]) for p in COVARIATE_NAMES])
        beta_chk = true_beta_for_regime(rg, np.random.default_rng(1000 + rep), nu,
                                        DEFAULT_SLOPES, t_range, sparse_shape=a.sparse_shape)
        if not np.allclose(beta, beta_chk):
            sys.exit(f"true beta at {rg} n={n} rep={rep} is not the regime calibration; "
                     f"check --sparse-shape")
        job = dict(regime=rg, n=int(n), rep=int(rep), nu=nu, beta=beta, rules=rules,
                   seed=int(cells[(rg, n)][rep].generate_state(1)[0]))
        X, t, alpha_t, Y = regenerate(job, t_range)
        # every rule's count, not only the entirely-NB ones, so a coincidental
        # match of one count cannot pass a wrong dataset
        for _, row in g.iterrows():
            nm = _norm_mask(row.partition, Y, alpha_t, X, t, a.threshold, nu,
                            a.kappa, X.shape[1] + 1)
            if int(nm.sum()) != int(row.n_norm):
                sys.exit(f"assignment mismatch at {rg} n={n} rep={rep} {row.partition}: "
                         f"regenerated n_norm={int(nm.sum())}, recorded {int(row.n_norm)}; "
                         f"--spawn-regimes, --seed, --threshold or --kappa differ from the sweep.")
        jobs.append(job)

    by = pd.Series([(j["regime"], j["n"]) for j in jobs]).value_counts().sort_index()
    print(f"[verify] {len(jobs)} datasets regenerate to the recorded Normal-assigned counts "
          f"for all four rules ({time.time() - t0:.0f} s)")
    for (rg, n), k in by.items():
        print(f"         {rg:10} n={n:<4} {k:5} datasets")
    if a.dry_run or not jobs:
        return

    # ---- fit, checkpointing every 10 datasets --------------------------------
    from joblib import Parallel, delayed
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} exists; pass --overwrite to replace it, or choose "
                         f"another --out")
    partial = a.out + ".partial"
    wrote_header = os.path.exists(partial)
    if wrote_header:
        check_resume(partial, tuple(a.cap), a.scan_spacing)
    rows, pending, done = [], [], 0
    t0 = time.time()
    gen = Parallel(n_jobs=a.n_jobs, verbose=5, return_as="generator")(
        delayed(fit_job)(j, a.steps, a.chains, t_range,
                          tuple(a.cap), a.scan_spacing)
        for j in jobs)
    for rr in gen:
        done += 1
        if rr:
            rows.extend(rr)
            pending.extend(rr)
        if pending and (done % 10 == 0 or done == len(jobs)):
            pd.DataFrame(pending).to_csv(partial, mode="a", index=False,
                                         header=not wrote_header)
            wrote_header, pending = True, []
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    ref = int(out.refused.notna().sum()) if "refused" in out else 0
    print(f"\n[fit] {len(jobs)} datasets in {(time.time() - t0) / 3600:.2f} h -> {a.out}; "
          f"refused rows {ref}")
    if ref:
        print(f"      refusals by reason: {out.refused.value_counts().to_dict()} "
              f"(mle_outside_cap should be 0)")
    if "capped_states" in out:
        fits = out[out.refused.isna()] if "refused" in out else out
        g = fits.groupby(["regime", "n", "rep"]).first()
        print(f"      states with I(u) reaching the cap: {int(g.capped_states.sum()):,} of "
              f"{int(g.states.sum()):,} (should be 0); split states "
              f"{int(g.split_states.sum()):,}; narrowest piece found by the scan alone over "
              f"the states held {g.min_state_unanchored_width.min():.3g} (spacing "
              f"{a.scan_spacing:g}); log-nu draws span [{g.lnu_draw_min.min():.2f}, "
              f"{g.lnu_draw_max.max():.2f}] against the cap ({a.cap[0]:g}, {a.cap[1]:g})")


if __name__ == "__main__":
    main()
