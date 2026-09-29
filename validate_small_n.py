"""Validation of both samplers against exact rejection draws from the definition
of the discrete GFD (Supplement Section 8.4, Table on n = 4).

For a small component (n <= 5, where rejection sampling is feasible) this script
  1. draws the exact reference: U* ~ Uniform(0,1)^n, kept when Q(y, U*) is
     nonempty, then V applied with log nu uniform on the exact feasible set and
     beta uniform on the polytope (exact_ref2.Reference; shares no code with the
     samplers);
  2. runs the single-site Gibbs sampler (discrete_gfi.DiscreteGFI) and reports its offset
     from the reference in reference-sd units and its KS distance;
  3. runs C independent chains of the theta-driven sampler (gfd_theta_mh.ThetaMH)
     with distinct seeds and reports the pooled offset with the between-chain
     standard error. At these n a single chain's effective sample size
     understates its error (Supplement Section 8.4), so no single-chain KS
     distance is reported for it.
  4. with --ref2 N, draws an independent second reference alongside the
     samplers; its offsets and KS distances from the first show how large
     those quantities are from reference noise alone. The Gibbs report gives
     the same yardstick from the chain's batch-means effective sample size, and
     all three report how often I(u) reaches the cap.

All three use the same selection rule, the same cap on log nu and the same box
|beta_k| <= b. The split-set fractions (I(u) with more than one piece) of the
reference's feasible u, the Gibbs sweeps and the MH states estimate the same
probability and are printed side by side.

Parallelism. The reference is drawn in --ref_chunks fixed chunks, chunk k with
its own generator default_rng([seed, 1, k]); the Gibbs chain and the C MH chains
(chain c seeded 1000 + 17 c + seed) then run as parallel tasks. Every random
stream is fixed by its seed alone, so the results do not depend on --n_jobs.

Datasets: --dataset n4 is the supplement's problem, y = (8, 27, 2, 0), K = 2
(generate_k2, seed 6); n3 is an intercept-only problem, y = (2, 14, 18)
(generate, seed 11). Others via --npz FILE (arrays X, y, t) or --generate n K seed.

    python validate_small_n.py --dataset n4 --ref 4000 --ref2 4000 --gibbs 20000 \
        --chains 16 --mh 16000 --n_jobs 48
"""
import argparse
import time

import numpy as np
from scipy.stats import ks_2samp

from discrete_gfi import DiscreteGFI, nb_mle
from exact_ref2 import Reference
from gfd_theta_mh import ThetaMH
from discrete_settings import (VALIDATION_CAP, VALIDATION_SCAN_SPACING,
                               REFERENCE_SCAN_SPACING)

# counters reported for each sampler
GIBBS_COUNTERS = ("U_updates", "skipped_updates", "reverted_sweeps", "capped", "box_hits",
                  "loo_multi", "multi_component", "anchor_misses", "narrow_pieces",
                  "narrow_pieces_loo", "center_retries", "beta_failures", "edge_disagreements")
GIBBS_WIDTHS = ("min_piece_width", "min_piece_width_loo", "min_unanchored_width")
MH_GEOM = ("vol_joggled", "anchor_infeasible", "anchor_missed_with_interior", "degenerate_width",
           "vol_fail_center", "vol_fail_qhull", "vol_fail_feasible", "edge_disagreements",
           "box_hits", "volume_calls")
MH_STATE = ("states", "split_states", "capped_states", "inf_theta", "inf_W")


def generate(n, K, seed):
    """Intercept plus K - 1 standard-normal covariates, beta = (1, 0.4, ...),
    nu = 0.3, exposures Uniform(0.5, 2)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, K - 1))])
    t = rng.uniform(0.5, 2.0, n)
    beta_true = np.concatenate([[1.0], 0.4 * np.ones(K - 1)])
    y = rng.negative_binomial(np.exp(X @ beta_true) * t, 0.3 / 1.3)
    return X, y.astype(float), t


def generate_k2(n, seed):
    """Intercept plus one standard-normal covariate, beta = (0.8, 0.4),
    nu = 0.3. With n = 4 and seed 6: y = (8, 27, 2, 0), x1 = (1.05, 1.78,
    -2.55, -0.14), t = (1.51, 0.99, 1.52, 0.68) to two decimals."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    t = rng.uniform(0.5, 2.0, n)
    y = rng.negative_binomial(np.exp(X @ np.array([0.8, 0.4])) * t, 0.3 / 1.3)
    return X, y.astype(float), t


DATASETS = {
    "n3": lambda: generate(3, 1, 11),
    "n4": lambda: generate_k2(4, 6),
}


# ---- tasks (module level so they can run in parallel) ------------------------

def _ref_chunk(X, y, t, a, lo, hi, k, need, stream=1):
    """Reference chunk k: up to `need` exact draws from its own generator.
    stream 1 is the reference; stream 2 is the independent second reference
    (--ref2), which shares no random numbers with it."""
    rng = np.random.default_rng([a.seed, stream, k])
    R = Reference(X, y, t, box=a.box, lo_cap=lo, hi_cap=hi)
    draws, tried, n_feas, n_split, n_capped, t0 = [], 0, 0, 0, 0, time.time()
    while len(draws) < need and time.time() - t0 < a.ref_budget:
        U = rng.uniform(size=len(y)); tried += 1
        d, pieces = R.draw(U, rng, scan_h=a.ref_scan)
        if pieces:
            n_feas += 1
            n_split += int(len(pieces) > 1)
            # a piece reaching the cap; Reference.intervals sets such an end
            # to the cap exactly
            n_capped += int(pieces[0][0] <= lo or pieces[-1][1] >= hi)
        if d is not None:
            draws.append(np.append(d[0], d[1]))
    return np.array(draws).reshape(-1, X.shape[1] + 1), tried, n_feas, n_split, n_capped


def _batch_se(x, batches=40):
    """Batch-means standard error of the mean of the series x (columns are
    separate series) and the implied effective sample size."""
    x = np.asarray(x, float)
    x = x.reshape(len(x), -1)
    batches = min(batches, len(x))              # short test runs
    if batches < 2:
        return np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
    m = len(x) // batches
    bm = x[:m * batches].reshape(batches, m, -1).mean(1)
    se = bm.std(0, ddof=1) / np.sqrt(batches)
    var = x.var(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(se > 0, var / se ** 2, np.nan)
    return se, ess


def _gibbs_chain(X, y, t, bh, nh, a, lo, hi):
    """The Gibbs chain: its draws (beta, log nu) from the sweeps that produced
    one, and its counters."""
    G = DiscreteGFI(X, y, t, bh, nh, seed=a.seed + 1, beta_box=a.box,
                    log_nu_bounds=(lo, hi), exact_ranges=True, scan_spacing=a.scan_spacing)
    out, capped = [], []
    from discrete_gfi import RESCALED           # live counts (see _mh_chain)
    r0 = dict(RESCALED)
    for _ in range(a.gibbs):
        before = getattr(G, "capped", 0)
        o = G.sweep()
        if o is not None:
            out.append(np.append(o[0], np.log(o[1])))
            # whether I(U) of this sweep reached the cap (the counter counts
            # capped pieces, so it can rise by two when both ends are capped)
            capped.append(int(getattr(G, "capped", 0) > before))
    counts = {k: getattr(G, k, None) for k in GIBBS_COUNTERS + GIBBS_WIDTHS}
    counts["capped_series"] = np.array(capped)
    counts.update({k: RESCALED[k] - r0[k] for k in RESCALED})
    return np.array(out).reshape(-1, X.shape[1] + 1), counts


def _mh_chain(X, y, t, bh, nh, c, a, lo, hi, h):
    """One theta-MH chain: its post-burn-in mean and its counters."""
    S = ThetaMH(X, y, t, bh, nh, seed=1000 + 17 * c + a.seed, log_nu_bounds=(lo, hi), scan_spacing=h)
    S.init()
    # imported here, not at the top: joblib copies a function defined in the
    # script into its worker with the globals it names, so a module-level
    # RESCALED would be a frozen copy rather than the sampler's live counts
    from discrete_gfi import RESCALED
    r0 = dict(RESCALED)                         # per-process counts: this chain's share
    burn = int(a.burn * a.mh)
    capped = []                                 # whether each state's I(U) reaches the cap

    def after_step(i):
        if i >= burn:
            capped.append(bool(S.state[3]["capped"]))

    D, _ = S.run(a.mh, progress=after_step)
    counts = {k: getattr(S.geom, k) for k in MH_GEOM}
    counts.update({k: getattr(S, k) for k in MH_STATE})
    counts["inf_theta_max_gap"] = S.inf_theta_max_gap
    counts["inf_theta_min_required_logv"] = S.inf_theta_min_required_logv
    counts["max_state_logv"] = S.max_state_logv
    counts["required_logv"] = list(S.inf_theta_required)
    counts["capped_frac"] = float(np.mean(capped)) if capped else np.nan
    counts["min_state_unanchored_width"] = S.min_state_unanchored_width
    counts.update({k: RESCALED[k] - r0[k] for k in RESCALED})
    return D[burn:].mean(0), counts, S.geom.min_unanchored_width


def _run(tasks, n_jobs):
    if n_jobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs)(delayed(f)(*args) for f, args in tasks)
    return [f(*args) for f, args in tasks]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dataset", choices=sorted(DATASETS), default="n4")
    src.add_argument("--npz", help="file with arrays X, y, t")
    src.add_argument("--generate", nargs=3, type=int, metavar=("n", "K", "seed"))
    ap.add_argument("--cap", nargs=2, type=float, default=list(VALIDATION_CAP),
                    help="bounds on log nu")
    ap.add_argument("--box", type=float, default=100.0, help="bound b on |beta_k|")
    ap.add_argument("--ref", type=int, default=4000, help="accepted reference draws")
    ap.add_argument("--ref2", type=int, default=0,
                    help="draws of an independent second reference, run alongside the "
                         "samplers; its offsets and KS distances from the first are the "
                         "yardstick for reference noise (0: none)")
    ap.add_argument("--ref_chunks", type=int, default=16,
                    help="the reference is drawn in this many chunks, each with its own "
                         "seed, so its draws do not depend on --n_jobs")
    ap.add_argument("--ref_budget", type=float, default=3600,
                    help="seconds allowed for each reference chunk")
    ap.add_argument("--gibbs", type=int, default=20000, help="Gibbs sweeps")
    ap.add_argument("--chains", type=int, default=16, help="independent theta-MH chains")
    ap.add_argument("--mh", type=int, default=16000, help="steps per theta-MH chain")
    ap.add_argument("--scan_spacing", type=float, default=VALIDATION_SCAN_SPACING,
                    help="log-nu scan spacing of the Gibbs sampler; a sweep costs "
                         "about (n + 1) * (cap width / spacing) slack LPs")
    ap.add_argument("--mh_scan_spacing", type=float, default=None,
                    help="log-nu scan spacing of the MH sampler (default: --scan_spacing). "
                         "Its cost is two scans per step, so a coarser value may be needed; "
                         "the narrowest piece found by the scan alone shows whether it sufficed")
    ap.add_argument("--ref_scan", type=float, default=REFERENCE_SCAN_SPACING,
                    help="log-nu scan spacing of the exact reference, finer than the "
                         "sampler's: the reference has no anchor, so a u whose whole "
                         "feasible set falls between scan points is rejected")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="parallel workers for the reference chunks and for the Gibbs "
                         "and MH chains; results do not depend on it")
    ap.add_argument("--burn", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.npz:
        z = np.load(a.npz); X, y, t = z["X"], z["y"].astype(float), z["t"]
    elif a.generate:
        X, y, t = generate(*a.generate)
    else:
        X, y, t = DATASETS[a.dataset]()
    n, K = X.shape
    lo, hi = a.cap
    mh_h = a.scan_spacing if a.mh_scan_spacing is None else a.mh_scan_spacing
    names = [f"beta{j}" for j in range(K)] + ["log nu"]
    print(f"n={n}, K={K}, y={y.astype(int).tolist()}, log nu in ({lo:g}, {hi:g}), b={a.box:g}")
    print(f"scan spacing: reference {a.ref_scan:g}, Gibbs {a.scan_spacing:g}, MH {mh_h:g}; "
          f"n_jobs {a.n_jobs}")
    bh, nh, _ = nb_mle(X, y, t)
    print(f"MLE: beta={bh.round(3).tolist()}, log nu={np.log(nh):.3f}\n", flush=True)

    # ---- exact reference, in fixed chunks ------------------------------------
    t0 = time.time()
    need = [a.ref // a.ref_chunks + (k < a.ref % a.ref_chunks) for k in range(a.ref_chunks)]
    chunks = _run([(_ref_chunk, (X, y, t, a, lo, hi, k, need[k])) for k in range(a.ref_chunks)],
                  a.n_jobs)
    ref = np.vstack([c[0] for c in chunks])
    tried, n_feas, n_split, n_capped = (sum(c[i] for c in chunks) for i in (1, 2, 3, 4))
    print(f"reference: {len(ref)} accepted of {tried:,} uniforms in {time.time() - t0:.0f}s")
    rm, rs = ref.mean(0), ref.std(0)
    print("  mean (sd): " + "  ".join(f"{nm} {m:.2f} ({s:.2f})" for nm, m, s in zip(names, rm, rs)))
    # Feasible u are exact draws from the uniform on the good set, the
    # stationary law of the Gibbs u-chain and the U-marginal of the MH target,
    # so the three split fractions estimate the same probability, and so do the
    # three capped fractions.
    ref_split = n_split / max(n_feas, 1)
    ref_capped = n_capped / max(n_feas, 1)
    print(f"  I(u) split in {n_split} of {n_feas} feasible u ({ref_split:.4f}, "
          f"se {np.sqrt(ref_split * (1 - ref_split) / max(n_feas, 1)):.4f})")
    print(f"  I(u) reaches the cap in {n_capped} of {n_feas} feasible u ({ref_capped:.4f}, "
          f"se {np.sqrt(ref_capped * (1 - ref_capped) / max(n_feas, 1)):.4f})", flush=True)

    # ---- Gibbs chain, MH chains and the second reference, in parallel ------------
    t0 = time.time()
    tasks = [(_gibbs_chain, (X, y, t, bh, nh, a, lo, hi))]
    tasks += [(_mh_chain, (X, y, t, bh, nh, c, a, lo, hi, mh_h)) for c in range(a.chains)]
    need2 = [a.ref2 // a.ref_chunks + (k < a.ref2 % a.ref_chunks) for k in range(a.ref_chunks)]
    tasks += [(_ref_chunk, (X, y, t, a, lo, hi, k, need2[k], 2))
              for k in range(a.ref_chunks) if need2[k] > 0]
    results = _run(tasks, a.n_jobs)
    secs = time.time() - t0
    gib_out, G = results[0]
    mh_results = results[1:1 + a.chains]
    ref2_chunks = results[1 + a.chains:]

    # ---- Gibbs report ------------------------------------------------------------
    gib = gib_out[int(a.burn * len(gib_out)):]
    print(f"\nGibbs: {a.gibbs} sweeps, "
          f"{len(gib_out)} usable, {G['reverted_sweeps']} restored, skipped updates "
          f"{G['skipped_updates']} (Gibbs and MH together took {secs:.0f}s)")
    n_loc = len(gib_out) + G["beta_failures"]
    g_split = G["multi_component"] / max(n_loc, 1)
    print(f"  split sets: leave-one-out {G['loo_multi']} of {G['U_updates']} updates, "
          f"I(U) {G['multi_component']} of {n_loc} sweeps ({g_split:.4f}; reference "
          f"{ref_split:.4f}, which it should match up to autocorrelation); anchor "
          f"misses {G['anchor_misses']}")
    print(f"  resolution: narrowest piece of I(U) {G['min_piece_width']:.3g} "
          f"({G['narrow_pieces']} narrower than the spacing); of a leave-one-out set "
          f"{G['min_piece_width_loo']:.3g} ({G['narrow_pieces_loo']} narrower); "
          f"narrowest piece found by the scan alone {G['min_unanchored_width']:.3g} "
          f"(should stay well above the spacing)")
    print(f"  numerical guards (all should be 0 or near it): skipped updates "
          f"{G['skipped_updates']}, reverted sweeps {G['reverted_sweeps']}, log-nu redraws "
          f"{G['center_retries']}, sweeps with no draw {G['beta_failures']}, box hits "
          f"{G['box_hits']}")
    print(f"  piece ends where the re-solve disagreed with the scan, the end then "
          f"taken at the last feasible test point (moves it by less than the "
          f"spacing): {G['edge_disagreements']}")
    print(f"  LPs re-solved at the set's own scale: Chebyshev centers "
          f"{G['center_resolved']} (first center off by over half its radius "
          f"{G['center_corrected']}; cold re-solves {G['center_cold_retry']}), slack values "
          f"near zero {G['slack_resolved']}")
    print("  offset (ref sd): " + "  ".join(f"{nm} {v:+.3f}" for nm, v in zip(names, (gib.mean(0) - rm) / rs)))
    print("  KS distance:     " + "  ".join(f"{nm} {ks_2samp(gib[:, j], ref[:, j]).statistic:.3f}"
                                         for j, nm in enumerate(names)))
    # noise yardsticks: the Gibbs draws' effective sample size (batch means) and
    # the approximate 95% point of the KS distance between the reference and an
    # independent sample of that size, 1.36 sqrt(1/n_ref + 1/ESS)
    g_se, g_ess = _batch_se(gib)
    print("  ESS (batch means): " + "  ".join(f"{nm} {e:.0f}" for nm, e in zip(names, g_ess)))
    print("  offset se (ref sd), reference and Gibbs combined: "
          + "  ".join(f"{nm} {np.sqrt(1 / len(ref) + (s / sd) ** 2):.3f}"
                      for nm, s, sd in zip(names, g_se, rs)))
    print("  KS 95% point from sampling noise: "
          + "  ".join(f"{nm} {1.36 * np.sqrt(1 / len(ref) + 1 / e):.3f}"
                      for nm, e in zip(names, g_ess)))
    cs = G["capped_series"][int(a.burn * len(G["capped_series"])):]
    if len(cs):
        c_se, _ = _batch_se(cs)
        print(f"  I(U) reaches the cap in {cs.mean():.4f} of sweeps after burn-in "
              f"(batch-means se {c_se[0]:.4f}; reference {ref_capped:.4f})")

    # ---- MH report -----------------------------------------------------------------
    means = (np.array([r[0] for r in mh_results]) - rm) / rs
    tot = {k: sum(r[1][k] for r in mh_results)
           for k in MH_GEOM + MH_STATE + ("center_resolved", "center_corrected", "slack_resolved",
                                    "center_cold_retry")}
    min_unanchored = min(r[2] for r in mh_results)
    print(f"\ntheta-MH: {a.chains} chains x {a.mh} steps")
    se = (means.std(0, ddof=1) / np.sqrt(a.chains) if a.chains > 1
          else np.full(means.shape[1], np.nan))
    print("  pooled offset (ref sd), between-chain se: "
          + "  ".join(f"{nm} {m:+.3f} ({s:.3f})" for nm, m, s in
                      zip(names, means.mean(0), se)))
    mh_split = tot["split_states"] / max(tot["states"], 1)
    print(f"  split sets: I(U) split in {tot['split_states']} of {tot['states']} states "
          f"({mh_split:.4f}; reference {ref_split:.4f}); capped states "
          f"{tot['capped_states']}; narrowest piece found by the scan alone "
          f"{min_unanchored:.3g} over all evaluations, "
          f"{min(r[1]['min_state_unanchored_width'] for r in mh_results):.3g} over the "
          f"states held (should stay well above the spacing)")
    cf = np.array([r[1]["capped_frac"] for r in mh_results])
    cf_se = cf.std(ddof=1) / np.sqrt(len(cf)) if len(cf) > 1 else np.nan
    print(f"  I(U) reaches the cap in {cf.mean():.4f} of states after burn-in "
          f"(between-chain se {cf_se:.4f}; reference {ref_capped:.4f})")
    max_gap = max(r[1]["inf_theta_max_gap"] for r in mh_results)
    need = min(r[1]["inf_theta_min_required_logv"] for r in mh_results)
    held = max(r[1]["max_state_logv"] for r in mh_results)
    print(f"  rejected because the target could not be evaluated: theta-moves "
          f"{tot['inf_theta']}, W-moves {tot['inf_W']} (should be 0)")
    print(f"    theta-moves: largest log acceptance ratio without its v factor {max_gap:.1f}; "
          f"to be accepted, the most favourable would have needed log v >= {need:.1f}, "
          f"against the largest log v any state held, {held:.1f} "
          f"(a margin of {need - held:.1f} nats; large and positive means harmless)")
    req = np.concatenate([r[1]["required_logv"] for r in mh_results] + [[]])
    if len(req):
        q = np.quantile(req, [0, 0.1, 0.5, 0.9, 1])
        print(f"    needed log v over all {len(req)}: min {q[0]:.1f}, 10% {q[1]:.1f}, "
              f"median {q[2]:.1f}, 90% {q[3]:.1f}, max {q[4]:.1f} (check_failed_proposals.py "
              f"recomputes a sample of these exactly; its range should cover this)")
    print(f"  numerical guards (all should be 0 or near it): anchor not in a piece "
          f"{tot['anchor_infeasible']} (with interior at the anchor, which would be a bug: "
          f"{tot['anchor_missed_with_interior']}), zero-length I(U) {tot['degenerate_width']}, "
          f"volume failures {tot['vol_fail_center'] + tot['vol_fail_qhull']} of "
          f"{tot['volume_calls']} (of which the set had interior "
          f"{tot['vol_fail_feasible']}), volumes computed only by joggling "
          f"{tot['vol_joggled']}, box hits {tot['box_hits']}")
    print(f"  LPs re-solved at the set's own scale: Chebyshev centers {tot['center_resolved']} "
          f"(first center off by over half its radius {tot['center_corrected']}; cold re-solves "
          f"after a failed warm start {tot['center_cold_retry']}), slack values near zero "
          f"{tot['slack_resolved']}")
    print(f"  piece ends where the re-solve disagreed with the scan, the end then taken at "
          f"the last feasible test point (moves it by less than the spacing): "
          f"{tot['edge_disagreements']} in {tot['volume_calls'] + tot['anchor_infeasible']} "
          f"evaluations")

    # ---- second reference: the yardstick for reference noise ---------------------
    if ref2_chunks:
        ref2 = np.vstack([c[0] for c in ref2_chunks])
        f2, s2, c2 = (sum(c[i] for c in ref2_chunks) for i in (2, 3, 4))
        print(f"\nsecond reference (independent seeds): {len(ref2)} draws")
        print("  offset from the reference (ref sd): "
              + "  ".join(f"{nm} {v:+.3f}" for nm, v in zip(names, (ref2.mean(0) - rm) / rs))
              + f"   (se {np.sqrt(1 / len(ref) + 1 / len(ref2)):.3f})")
        print("  KS distance from the reference:     "
              + "  ".join(f"{nm} {ks_2samp(ref2[:, j], ref[:, j]).statistic:.3f}"
                          for j, nm in enumerate(names))
              + f"   (95% point {1.36 * np.sqrt(1 / len(ref) + 1 / len(ref2)):.3f})")
        print(f"  I(u) split in {s2} and reaches the cap in {c2} of {f2} feasible u "
              f"({c2 / max(f2, 1):.4f})")


if __name__ == "__main__":
    main()
