"""Frequentist coverage of the discrete generalized fiducial distribution.

The sparse regime of Section 5: every observation is NB, so the discrete
construction applies, computed with the MH sampler (gfd_theta_mh.py). Each
replication is also fit with the flat-prior posterior on the same data (an
independence sampler on the NB likelihood), for the comparison of Figure 3 and
Supplement Section 8.5.

With --shape the intercept is calibrated to that geometric-mean NB shape (the
paper's sparse regime is shape 5), so nu sets the mean. Replications are seeded
by default_rng([seed, rep]), so --reps shards reproduce the full run exactly.
Every row records the cap, the scan spacing and the sampler's guard counters.

    python sim_discrete_coverage.py --R 2000 --n 50 --nu 0.05 --shape 5 --n_jobs 48
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrete_settings import SIM_CAP, SIM_SCAN_SPACING  # noqa: E402


def ess(x):
    x = np.asarray(x, float); x = x - x.mean(); m = len(x)
    ac = np.correlate(x, x, "full")[m - 1:] / (x @ x)
    k = int(np.argmax(ac < 0.05)) or m - 1
    return m / (1 + 2 * ac[1:k].sum())


def dataset_base(rep, X, y):
    """Per-dataset descriptors recorded on every output row."""
    return dict(rep=rep, counts_med=float(np.median(y)), zeros=int((y == 0).sum()),
                pos_rank=int(np.linalg.matrix_rank(X[y > 0])) if (y > 0).any() else 0,
                K=X.shape[1])


def one(rep, n, nu_true, beta_true, steps, seed, chains=1,
        cap=SIM_CAP, scan_spacing=SIM_SCAN_SPACING):
    """Generate one dataset from this script's own seeding, then fit it."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sim_coverage import make_synthetic_design
    rng = np.random.default_rng([seed, rep])
    X, t = make_synthetic_design(n, rng)
    y = rng.negative_binomial(np.exp(X @ beta_true) * t, nu_true / (1 + nu_true)).astype(float)
    truth = np.append(beta_true, np.log(nu_true))
    # The flat comparator continues on the same Generator the data came from.
    return fit_dataset(X, y, t, truth, dataset_base(rep, X, y), steps,
                       chain_seed=seed + rep, chains=chains,
                       rng_flat=rng, cap=cap, scan_spacing=scan_spacing)


def flat_posterior_draws(X, y, t, S, rng, n_draws=20000, burn=4000):
    """Draws from the flat-prior posterior on (beta, log nu): independence
    Metropolis-Hastings on the NB likelihood, with the proposal S.prop of the
    ThetaMH sampler S (the multivariate t at the MLE). Shared by the coverage
    scripts so the comparator is the same everywhere."""
    from gfd_theta_mh import nb_loglik
    th = S.theta_hat.copy(); lp = nb_loglik(X, y, t, th[:-1], th[-1]) - S.prop.logpdf(th)
    F = np.empty((n_draws, X.shape[1] + 1))
    for i in range(n_draws):
        c = S.prop.rvs(random_state=rng)
        lc = nb_loglik(X, y, t, c[:-1], c[-1]) - S.prop.logpdf(c)
        if np.log(rng.uniform()) < lc - lp:
            th, lp = c, lc
        F[i] = th
    return F[burn:]


def fit_dataset(X, y, t, truth, base, steps, chain_seed, chains=1,
                rng_flat=None, cap=SIM_CAP, scan_spacing=SIM_SCAN_SPACING):
    """Fit the discrete GFD to a given dataset.

    Datasets generated elsewhere -- in particular sim_coverage.py's, which
    Table 2 pairs across partition rules -- are fit through this function with
    the same sampler and output schema as one(). Chain c is seeded
    chain_seed + 100_003 * c. rng_flat drives the independence-sampler flat
    comparator; pass None to skip it (flat_* columns come back NaN), as the
    partition driver does, since there the Stan flat posterior is the comparator.

    `cap` bounds log nu and `scan_spacing` is the spacing of the scan that
    locates the pieces of I(U) (gfd_theta_mh); both are recorded on every row.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from discrete_gfi import nb_mle
    from gfd_theta_mh import ThetaMH
    lo_cap, hi_cap = map(float, cap)
    # every refusal is recorded with its reason: dropping them silently would
    # estimate coverage on the well-behaved datasets only, which is a conditional
    # statement and looks better than the unconditional one
    try:
        bh, nh, _ = nb_mle(X, y, t)
    except Exception as e:
        return [dict(base, refused="mle_" + type(e).__name__)]
    if not lo_cap < np.log(nh) < hi_cap:
        return [dict(base, refused="mle_outside_cap")]
    # C independent chains with distinct seeds, pooled after burn-in; with C > 1
    # the between-chain spread of the means (chain_sd) gives the Monte Carlo error.
    from discrete_gfi import RESCALED
    r0 = dict(RESCALED)                 # per-process counts: keep this fit's share
    t0 = time.time()
    Ds, samplers = [], []
    for c in range(chains):
        try:
            S = ThetaMH(X, y, t, bh, nh, seed=chain_seed + 100_003 * c, log_nu_bounds=(lo_cap, hi_cap),
                        scan_spacing=scan_spacing)
            S.init()
        except Exception as e:
            return [dict(base, refused="init_" + type(e).__name__)]
        Dc = np.empty((steps, X.shape[1] + 1))
        for i in range(steps):
            S.step(); Dc[i] = S.state[0]
        Ds.append(Dc[steps // 5:]); samplers.append(S)
    D = np.vstack(Ds)
    chain_means = np.array([Dc.mean(0) for Dc in Ds])            # (chains, K+1)
    geom_sum = lambda attr: int(sum(getattr(s.geom, attr) for s in samplers))
    chain_sum = lambda attr: int(sum(getattr(s, attr) for s in samplers))
    unbounded_any = bool(any(s.geom.unbounded_by_design for s in samplers))
    # Diagnostics recorded on every row: the numerical guards of gfd_theta_mh
    # (all should be 0 or near it), the split and capped states, and the
    # narrowest piece the scan found without an anchor, to compare with the
    # spacing.
    diag = dict(cap_lo=lo_cap, cap_hi=hi_cap, scan_spacing=scan_spacing,
                states=chain_sum("states"), split_states=chain_sum("split_states"),
                capped_states=chain_sum("capped_states"),
                inf_theta=chain_sum("inf_theta"), inf_W=chain_sum("inf_W"),
                anchor_infeasible=geom_sum("anchor_infeasible"),
                anchor_missed_with_interior=geom_sum("anchor_missed_with_interior"),
                vol_fail_feasible=geom_sum("vol_fail_feasible"),
                edge_disagreements=geom_sum("edge_disagreements"),
                vol_joggled=geom_sum("vol_joggled"),
                **{k: RESCALED[k] - r0[k] for k in RESCALED},
                min_unanchored_width=float(min(s.geom.min_unanchored_width for s in samplers)),
                # the same over the states held only (the above includes
                # rejected proposals), and the range of the log-nu draws,
                # which shows how far inside the cap the distribution lies
                min_state_unanchored_width=float(min(s.min_state_unanchored_width
                                                     for s in samplers)),
                lnu_draw_min=float(D[:, -1].min()), lnu_draw_max=float(D[:, -1].max()))
    # flat-prior comparator on the same data
    F = flat_posterior_draws(X, y, t, S, rng_flat) if rng_flat is not None else None
    rows = []
    for j, nm in enumerate(["beta0", "beta1", "beta2", "beta3", "log_nu"]):
        gl, gu = np.quantile(D[:, j], [.025, .975])
        fl, fu = (np.quantile(F[:, j], [.025, .975]) if F is not None
                  else (np.nan, np.nan))
        rows.append(dict(base, param=nm, truth=truth[j], chains=chains,
                         unbounded=unbounded_any,
                         box_hits=geom_sum("box_hits"),
                         degenerate=geom_sum("degenerate_width"),
                         vol_calls=geom_sum("volume_calls"),
                         **diag,
                         gfd_cov=int(gl <= truth[j] <= gu), gfd_len=gu - gl,
                         # Endpoints, not only the width: the log-nu interval
                         # has to be carried to the nu scale for comparison with
                         # sweeps that report nu, and exp() of a width is not a
                         # width. Quantiles are equivariant, so the nu-scale
                         # interval is exactly [exp(gfd_lo), exp(gfd_hi)].
                         gfd_lo=gl, gfd_hi=gu, gfd_median=float(np.median(D[:, j])),
                         gfd_mean=D[:, j].mean(), gfd_ess=ess(D[:, j]),
                         chain_sd=float(chain_means[:, j].std(ddof=1)) if chains > 1 else np.nan,
                         flat_cov=int(fl <= truth[j] <= fu) if F is not None else np.nan,
                         flat_len=fu - fl, flat_lo=fl, flat_hi=fu,
                         flat_mean=F[:, j].mean() if F is not None else np.nan,
                         shift_sd=((D[:, j].mean() - F[:, j].mean()) / F[:, j].std()
                                   if F is not None else np.nan),
                         secs=time.time() - t0, acc=S.acc["theta"] / S.tries["theta"]))
    return rows


def check_resume(partial, cap, scan_spacing):
    """A run resumes by appending to <out>.partial. Refuse if that file was
    written with a different cap or scan spacing, which would silently merge
    fits from two settings."""
    prev = pd.read_csv(partial)
    cols = ["cap_lo", "cap_hi", "scan_spacing"]
    if not set(cols) <= set(prev.columns):
        raise SystemExit(f"{partial} records no cap or scan spacing (an older run); move it "
                         f"aside before starting this one")
    seen = prev[cols].dropna().drop_duplicates().to_numpy()
    want = np.array([[cap[0], cap[1], scan_spacing]], float)
    if len(seen) and not (len(seen) == 1 and np.allclose(seen, want)):
        raise SystemExit(f"{partial} was written with cap/scan spacing {seen.tolist()}, not "
                         f"{want.tolist()[0]}; move it aside or rerun with its settings")
    print(f"resuming: appending to {partial} ({prev.rep.nunique()} replications already there)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--reps", type=int, nargs="+", default=None,
                    help="run only these replication indices (same seeds as the full run, "
                         "so results pair with it)")
    ap.add_argument("--chains", type=int, default=1,
                    help="independent chains per replication, pooled; use >1 when "
                         "ESS(log nu)/ESS(beta) is well below 1 (sparse, zero-heavy data)")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--shape", type=float, default=5.0,
                    help="geometric-mean NB shape the intercept is calibrated to")
    ap.add_argument("--cap", nargs=2, type=float, default=list(SIM_CAP),
                    help="bounds on log nu (part of the parameter set Theta)")
    ap.add_argument("--scan_spacing", type=float, default=SIM_SCAN_SPACING,
                    help="log-nu spacing of the scan that locates the pieces of I(u); "
                         "each step costs about 2 * (cap width / spacing) slack LPs")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing results file")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sim_coverage import beta_for_shape
    beta = beta_for_shape(a.shape, np.random.default_rng(a.seed))
    out = a.out or f"sim_discrete_coverage_shape{a.shape:g}_n{a.n}_nu{a.nu}.csv"
    print(f"discrete GFD coverage: R={a.R}, n={a.n}, nu={a.nu}, {a.steps} steps x {a.chains} chain(s), "
          f"cap={tuple(a.cap)}, scan spacing {a.scan_spacing:g}, "
          f"reps={'all' if not a.reps else len(a.reps)}, beta={beta}", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    # Replications are appended to <out>.partial as they finish, so an
    # interrupted run resumes with --reps on the missing ones (check_resume).
    reps_to_run = list(a.reps if a.reps else range(a.R))
    if os.path.exists(out) and not a.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it, or choose "
                         f"another --out")
    partial = out + ".partial"
    # An existing partial file already carries the header, so a resumed run
    # appends to it rather than writing a second one mid-file.
    rows, pending, done = [], [], 0
    wrote_header = os.path.exists(partial)
    if wrote_header:
        check_resume(partial, tuple(a.cap), a.scan_spacing)
    gen = Parallel(n_jobs=a.n_jobs, verbose=5, return_as="generator")(
        delayed(one)(r, a.n, a.nu, beta, a.steps, a.seed, a.chains,
                     tuple(a.cap), a.scan_spacing)
        for r in reps_to_run)
    for rr in gen:
        done += 1
        if rr:
            new = rr if isinstance(rr, list) else [rr]
            rows.extend(new)
            pending.extend(new)
        if pending and (done % 10 == 0 or done == len(reps_to_run)):
            pd.DataFrame(pending).to_csv(partial, mode="a", index=False,
                                         header=not wrote_header)
            wrote_header, pending = True, []
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if "refused" not in df:
        df["refused"] = np.nan
    ref = df[df.refused.notna()]
    ok = df[df.refused.isna()]
    R_run = len(a.reps) if a.reps else a.R
    print(f"\n{len(ok)//5} replications of {R_run} in {(time.time()-t0)/3600:.2f} h -> {out}")
    print(f"refused by the guards: {len(ref)} of {R_run} ({100*len(ref)/R_run:.1f}%)"
          + ("" if ref.empty else "  " + str(ref.refused.value_counts().to_dict())))
    if not ok.empty:
        print(f"datasets with a zero count: {100*(ok.groupby('rep').zeros.first()>0).mean():.1f}%; "
              f"positive-count rows rank-deficient: "
              f"{100*(ok.groupby('rep').pos_rank.first() < ok.K.iloc[0]).mean():.1f}%; "
              f"box-truncated (unbounded_by_design): "
              f"{100*ok.groupby('rep').unbounded.first().mean():.1f}%; "
              f"box hits > 0: {100*(ok.groupby('rep').box_hits.first()>0).mean():.1f}%; "
              f"zero-width feasible sets: {int(ok.groupby('rep').degenerate.first().sum())} "
              f"in {int(ok.groupby('rep').vol_calls.first().sum()):,} evaluations")
    print()
    print(f"{'param':8} {'GFD cov':>9} {'flat cov':>9} {'GFD len':>9} {'flat len':>9} "
          f"{'len ratio':>10} {'shift (sd)':>11} {'ESS':>7}")
    for p in ["beta0", "beta1", "beta2", "beta3", "log_nu"]:
        s = ok[ok.param == p]
        if s.empty:
            continue
        se = np.sqrt(s.gfd_cov.mean() * (1 - s.gfd_cov.mean()) / len(s))
        print(f"{p:8} {s.gfd_cov.mean():>9.3f} {s.flat_cov.mean():>9.3f} "
              f"{s.gfd_len.mean():>9.4f} {s.flat_len.mean():>9.4f} "
              f"{(s.gfd_len/s.flat_len).mean():>10.4f} {s.shift_sd.mean():>+11.4f} "
              f"{s.gfd_ess.median():>7.0f}   (se {se:.3f})")
    if "capped_states" in ok:
        g = ok.groupby("rep").first()
        print(f"states with I(u) reaching the cap: {int(g.capped_states.sum()):,} of "
              f"{int(g.states.sum()):,} (should be 0; refusals mle_outside_cap: "
              f"{int((ref.refused == 'mle_outside_cap').sum())}); split states: {int(g.split_states.sum()):,}; narrowest "
              f"piece found by the scan alone {g.min_unanchored_width.min():.3g} "
              f"(scan spacing {g.scan_spacing.iloc[0]:g}); over the states held "
              f"{g.min_state_unanchored_width.min():.3g}")
        print(f"log-nu draws span [{g.lnu_draw_min.min():.2f}, {g.lnu_draw_max.max():.2f}] "
              f"against the cap ({g.cap_lo.iloc[0]:g}, {g.cap_hi.iloc[0]:g}); seconds per "
              f"replication divided by steps (the flat-prior comparator included): median "
              f"{np.median(g.secs / a.steps):.3f}, max {np.max(g.secs / a.steps):.3f}")
        print(f"numerical guards (should be 0 or near it): theta proposals rejected as "
              f"unevaluable {int(g.inf_theta.sum()):,}, W proposals {int(g.inf_W.sum())}, "
              f"anchor missed with interior {int(g.anchor_missed_with_interior.sum())}, volume "
              f"failures with interior {int(g.vol_fail_feasible.sum())}, piece-end "
              f"disagreements {int(g.edge_disagreements.sum())}")
    ess_ratio = ok[ok.param == "log_nu"].gfd_ess.values / ok[ok.param == "beta0"].gfd_ess.values
    print(f"ESS(log nu)/ESS(beta0): median {np.median(ess_ratio):.2f}; replications with ESS(log nu) < 50: "
          f"{int((ok[ok.param == 'log_nu'].gfd_ess < 50).sum())} of {len(ok)//5}  "
          f"(below ~0.8 a single chain can understate its error; use --chains > 1)")
    if "chain_sd" in ok and ok.chain_sd.notna().any():
        print("between-chain sd of the mean vs the pooled ESS-based se (ratio ~1 means the ESS is honest):")
        for p in ["beta0", "log_nu"]:
            s = ok[ok.param == p]
            se_ess = (s.gfd_len / 3.92) / np.sqrt(s.gfd_ess)      # interval length -> sd, / sqrt(ESS)
            print(f"  {p:7} median chain sd {s.chain_sd.median():.4f}, median ESS se {se_ess.median():.4f}, "
                  f"ratio {np.median(s.chain_sd / se_ess * np.sqrt(s.chains)):.2f}")
    clean = ok[~ok.unbounded.astype(bool)] if "unbounded" in ok else ok
    if len(clean) < len(ok):
        print("\ncoverage among replications where the box does not bind "
              f"({len(clean)//5} of {len(ok)//5}):")
        for p in ["beta0", "beta1", "beta2", "beta3", "log_nu"]:
            s = clean[clean.param == p]
            if not s.empty:
                print(f"  {p:8} GFD {s.gfd_cov.mean():.3f}  flat {s.flat_cov.mean():.3f}")


if __name__ == "__main__":
    main()
