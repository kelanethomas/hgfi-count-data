"""High-precision check of the theta proposals the MH sampler rejected because
their target could not be evaluated (Supplement, Section 8.3, Numerical guards).

Reruns the validation's MH chains (same seeds: chain c seeded 1000 + 17 c +
seed; n = 4 problem, cap (-6, 6)), records the failed proposals, and recomputes
for each the cells, u, the eta windows, the pieces of I(u), the polytope volume
and both log-likelihoods in 50-digit arithmetic (mpmath), giving its true log v
and true acceptance probability. The proposal ratio is evaluated in double
precision and log v of the current state is the sampler's own value.

    python check_failed_proposals.py --chains 16 --steps 16000 --below 100 --n_jobs 16

--below recomputes every failed proposal whose needed log v is below that value
(the ones closest to acceptance).
"""
import argparse
import time

import mpmath as mp
import numpy as np

from discrete_gfi import feasible_pieces, nb_mle
from gfd_theta_mh import ThetaMH
from validate_small_n import generate_k2

mp.mp.dps = 50
LOGR_LO, LOGR_HI, ITERS = -200.0, 60.0, 64      # bisection on log R; width 260 / 2^64


def F(y, r, p):
    """NB distribution function F(y; r, p) = I_p(r, y + 1), in high precision."""
    if y < 0:
        return mp.mpf(0)
    return mp.betainc(r, y + 1, 0, p, regularized=True)


def solve_logR(u, y, p):
    """log R with F(y; R, p) = u, by bisection (F is decreasing in R)."""
    lo, hi = mp.mpf(LOGR_LO), mp.mpf(LOGR_HI)
    for _ in range(ITERS):
        mid = (lo + hi) / 2
        if F(y, mp.e ** mid, p) > u:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def loglik_mp(X, y, t, theta):
    """log L(theta) = sum_i log(F(y_i) - F(y_i - 1)), in high precision."""
    beta, nu = theta[:-1], mp.e ** mp.mpf(float(theta[-1]))
    p = nu / (1 + nu)
    ll = mp.mpf(0)
    for i in range(len(y)):
        r = mp.e ** mp.mpf(float(X[i] @ beta)) * mp.mpf(float(t[i]))
        ll += mp.log(F(int(y[i]), r, p) - F(int(y[i]) - 1, r, p))
    return float(ll)


def make_bounds_fn(u, y, t):
    """eta bounds (lower, upper), shape (S, n), for high-precision uniforms u."""
    def bounds_fn(lnus):
        lnus = np.atleast_1d(lnus)
        lower = np.full((len(lnus), len(y)), -np.inf)
        upper = np.empty((len(lnus), len(y)))
        for s, lam in enumerate(lnus):
            nu = mp.e ** mp.mpf(float(lam))
            p = nu / (1 + nu)
            for i in range(len(y)):
                upper[s, i] = float(solve_logR(u[i], int(y[i]), p) - mp.log(t[i]))
                if y[i] > 0:
                    lower[s, i] = float(solve_logR(u[i], int(y[i]) - 1, p) - mp.log(t[i]))
        return lower, upper
    return bounds_fn


def exact_check(S, fail, scan):
    """True log v, log acceptance ratio and diagnostics for one failed proposal."""
    X, y, t, geom = S.X, S.y, S.t, S.geom
    th, W = fail["theta_new"], fail["W"]
    beta, lam = th[:-1], float(th[-1])
    nu = mp.e ** mp.mpf(lam)
    p = nu / (1 + nu)
    u, one_minus_u, cell_width = [], [], []
    for i in range(len(y)):
        r = mp.e ** mp.mpf(float(X[i] @ beta)) * mp.mpf(float(t[i]))
        lo, hi = F(int(y[i]) - 1, r, p), F(int(y[i]), r, p)
        u.append(lo + mp.mpf(float(W[i])) * (hi - lo))
        one_minus_u.append(float(mp.log10(1 - u[-1])) if u[-1] < 1 else -np.inf)
        cell_width.append(float(mp.log10(hi - lo)) if hi > lo else -np.inf)
    bounds_fn = make_bounds_fn(u, y, t)
    lo_eta, hi_eta = (b[0] for b in bounds_fn(np.array([lam])))
    windows = hi_eta - lo_eta                   # widths of the X_i' beta windows at lam
    pieces, _ = feasible_pieces(None, y, t, geom.slack_lp, scan, anchors=[lam],
                                bounds_fn=bounds_fn)
    total = sum(c - a for a, c in pieces)
    held = any(a - 1e-12 <= lam <= c + 1e-12 for a, c in pieces)
    lv_vol = geom.log_volume(None, lam, bounds=(lo_eta, hi_eta))
    ll_new, ll_cur = loglik_mp(X, y, t, th), loglik_mp(X, y, t, fail["theta_cur"])
    out = dict(log10_1_minus_u=one_minus_u, log10_cell_width=cell_width,
               eta_windows=windows, n_pieces=len(pieces), I_len=total, anchor_in_piece=held,
               required_logv=fail["required_logv"],
               loglik_err=max(abs(ll_new - fail["loglik_new"]), abs(ll_cur - fail["loglik_cur"])))
    if not held or total <= 0 or not np.isfinite(lv_vol):
        out.update(true_logv=np.nan, log_accept=np.nan)
        return out
    true_logv = -np.log(total) - lv_vol
    log_accept = ll_new + true_logv - (ll_cur + fail["logv_cur"]) + fail["log_k"]
    out.update(true_logv=true_logv, log_accept=log_accept)
    return out


def _chain(c, a, X, y, t, bh, nh, scan, echo=False):
    """Chain c of the validation (same seed): run it, recompute its failed
    proposals (those needing log v below --below, if given), and return the
    printed lines, the rows and the needed log v of every failure."""
    # set here as well as at import: a parallel worker receives this function
    # without running the module's top level, and would otherwise compute at
    # mpmath's default 15 digits
    mp.mp.dps = 50
    lines = []

    def say(msg):
        lines.append(msg)
        if echo:
            print(msg, flush=True)

    lo, hi = a.cap
    keep_all = np.isfinite(a.below)
    S = ThetaMH(X, y, t, bh, nh, seed=1000 + 17 * c + a.seed, log_nu_bounds=(lo, hi), scan_spacing=0.02,
                record_failures=10 ** 6 if keep_all else a.max_failures)
    S.init()
    t0 = time.time()
    S.run(a.steps)
    fails = [f for f in S.failures if f["required_logv"] < a.below]
    say(f"chain {c}: {a.steps} steps, {S.inf_theta} theta proposals rejected as "
        f"unevaluable, {len(fails)} recomputed, largest log v held {S.max_state_logv:.1f} "
        f"({time.time() - t0:.0f}s)")
    rows = []
    for k, fail in enumerate(fails):
        t0 = time.time()
        r = exact_check(S, fail, scan)
        r.update(chain=c, k=k, max_state_logv=S.max_state_logv)
        rows.append(r)
        ok = np.isfinite(r["log_accept"])
        say(f"  failure {k:2d}: min log10(1-u) {min(r['log10_1_minus_u']):7.1f}, "
            f"min log10(cell width) {min(r['log10_cell_width']):7.1f} (double precision "
            f"underflows below -308), narrowest window {np.min(r['eta_windows']):.3g}, "
            f"pieces {r['n_pieces']}, needed log v {r['required_logv']:6.1f}, "
            + (f"true log v {r['true_logv']:7.2f} (largest held {S.max_state_logv:.2f}), "
               f"true log acceptance ratio {r['log_accept']:8.1f}"
               if ok else "still not evaluable in high precision")
            + f"  [{time.time() - t0:.0f}s]")
    return lines, rows, list(S.inf_theta_required)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000, help="steps per chain")
    ap.add_argument("--max_failures", type=int, default=30, help="per chain")
    ap.add_argument("--scan", type=float, default=0.05,
                    help="log-nu scan spacing for the high-precision pieces of I(u)")
    ap.add_argument("--cap", nargs=2, type=float, default=[-6.0, 6.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--below", type=float, default=np.inf,
                    help="recompute only the failed proposals whose needed log v is below "
                         "this (all of them, however many; --max_failures then does not apply)")
    ap.add_argument("--n_jobs", type=int, default=1, help="chains run in parallel")
    a = ap.parse_args()

    X, y, t = generate_k2(4, 6)
    bh, nh, _ = nb_mle(X, y, t)
    lo, hi = a.cap
    scan = np.linspace(lo, hi, int(np.ceil((hi - lo) / a.scan)) + 1)
    tasks = [(c, a, X, y, t, bh, nh, scan) for c in range(a.chains)]
    if a.n_jobs > 1:
        from joblib import Parallel, delayed
        # the workers import _chain from this module by name: a function
        # defined in __main__ would be pickled whole, with what it references,
        # which fails on some Python/joblib versions
        from check_failed_proposals import _chain as chain_fn
        results = Parallel(n_jobs=a.n_jobs)(delayed(chain_fn)(*tk) for tk in tasks)
    else:
        results = [_chain(*tk, echo=True) for tk in tasks]
    rows, all_required = [], []
    for lines, r_rows, req in results:
        if a.n_jobs > 1:
            print("\n".join(lines), flush=True)
        rows += r_rows
        all_required += req

    ok = [r for r in rows if np.isfinite(r["log_accept"])]
    print(f"\n{len(rows)} failed proposals recomputed; {len(ok)} evaluable in 50-digit precision")
    if ok:
        la = np.array([r["log_accept"] for r in ok])
        tv = np.array([r["true_logv"] - r["max_state_logv"] for r in ok])
        w = np.array([np.min(r["eta_windows"]) for r in ok])
        print(f"  true log acceptance ratio: largest {la.max():.1f}, median {np.median(la):.1f} "
              f"(acceptance probability at most {np.exp(min(la.max(), 0)):.3g})")
        print(f"  true log v minus the largest log v the chain held: largest {tv.max():.2f}")
        print(f"  narrowest eta window: smallest {w.min():.3g}, median {np.median(w):.3g} "
              f"(ordinary windows mean the failures were rounding, not thin sets)")
    if rows:
        req = np.array([r["required_logv"] for r in rows])
        allr = np.array(all_required)
        print(f"  needed log v, recomputed sample: {req.min():.1f} to {req.max():.1f}; all "
              f"{len(allr)} unevaluable proposals in these chains: {allr.min():.1f} to "
              f"{allr.max():.1f} (the smallest, the most favourable, "
              f"{'is' if req.min() <= allr.min() else 'is NOT'} in the sample)")
        print(f"  largest error of the sampler's double-precision log-likelihoods: "
              f"{max(r['loglik_err'] for r in rows):.2g}")
    bad = [r for r in rows if not np.isfinite(r["log_accept"])]
    if bad:
        print(f"  {len(bad)} still not evaluable: these need a closer look")


if __name__ == "__main__":
    main()
