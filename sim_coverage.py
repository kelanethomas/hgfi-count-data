#!/usr/bin/env python3
"""Coverage simulation for the single-component hybrid model (Section 5).

Each replication draws a fresh i.i.d. design (make_synthetic_design: intercept,
a Uniform(-sqrt 3, sqrt 3) covariate, two indicators of a three-level factor with
probabilities (0.40, 0.30, 0.30), exposure Uniform(60, 90)) and exact NB data
    Y_i ~ NB(r_i = exp(X_i' beta) t_i, nu / (1 + nu)),  beta = (beta_0, 0.10, 0.20, -0.15),
then fits, on the same data, every partition rule in --partition and every
method in --methods:
    gfd         model_single_bin_hybrid.stan (likelihood times Jacobian);
                model_bin_nb.stan when nothing is assigned to the Normal component
                (the paper replaces those fits with the discrete GFD:
                sim_fill_rank_guard.py, sim_partition_discrete.py)
    bayes_flat  model_single_bin_hybrid_flat.stan (flat prior on (beta, log nu))
Partition rules (_raw_norm_mask): plugin_rate (the mean rule, tau = --threshold),
expected_shape (oracle shape rule, omega = --omega), plugin_shape, plugin_shape_lcb,
and realized (the observation's own count). Under every rule a fit left with
fewer than K + 1 Normal-assigned observations is entirely NB.

The intercept sets the regime: geometric-mean count 3000 (heavy) or 500
(borderline), or geometric-mean shape --sparse-shape (sparse). With --nu-range,
nu is drawn log-uniformly per replication and the intercept recalibrated.

Output: one row per (regime, n, rep, method, partition, parameter), appended
after each (regime, n) cell; failed fits are rows with param "__fit_error__".

    python sim_coverage.py --regimes heavy borderline --R 2000 --n_jobs 48
    python sim_coverage.py --regimes heavy borderline --nu 0.005 --R 1000 ...
    python sim_coverage.py --nu-range 0.002,0.07 --partition expected_shape \
        plugin_rate plugin_shape plugin_shape_lcb --methods gfd --R 1000 ...
"""

import os
import zlib
import argparse
import logging
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / environment
# --------------------------------------------------------------------------- #
THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
STAN_FILE      = os.path.join(THIS_DIR, "model_single_bin_hybrid.stan")
STAN_FILE_FLAT = os.path.join(THIS_DIR, "model_single_bin_hybrid_flat.stan")
STAN_FILE_NB   = os.path.join(THIS_DIR, "model_bin_nb.stan")

COVARIATE_NAMES  = ["Intercept", "X_cont", "X_grpA", "X_grpB"]
DEFAULT_SLOPES   = np.array([0.10, 0.20, -0.15])     # X_cont, X_grpA, X_grpB
DEFAULT_NU       = 0.05
REGIME_TARGET_MU = {"heavy": 3000.0, "borderline": 500.0}   # sparse: see --sparse-shape
REGIMES          = ["heavy", "borderline", "sparse"]
DEFAULT_N_GRID   = [50, 100, 200, 500]
GROUP_PROBS      = np.array([0.40, 0.30, 0.30])      # reference level, grpA, grpB
T_RANGE          = (60.0, 90.0)                      # exposure bounds


# --------------------------------------------------------------------------- #
# Synthetic design generator (fresh iid draw each replication)
# --------------------------------------------------------------------------- #
def make_synthetic_design(n, rng, t_range=T_RANGE):
    """Fresh iid design of size n: intercept, one bounded mean-zero,
    unit-variance continuous covariate, two indicators for a 3-level
    categorical covariate, and bounded exposure t."""
    x_cont = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n)
    grp = rng.choice(3, size=n, p=GROUP_PROBS)        # 0=ref, 1=grpA, 2=grpB
    X = np.column_stack([np.ones(n), x_cont,
                         (grp == 1).astype(float),
                         (grp == 2).astype(float)])
    t = rng.uniform(*t_range, size=n)
    return X, t


def beta_for_shape(target_shape, rng, slopes=DEFAULT_SLOPES, t_range=T_RANGE):
    """beta with the intercept set so the population geometric-mean NB shape
    exp(x' beta) t is target_shape (from a large draw of the design); the shape
    does not involve nu, so nu then sets the mean."""
    Xp, tp = make_synthetic_design(200_000, rng, t_range)
    mean_slope = float(np.mean(Xp[:, 1:] @ slopes))
    mean_log_t = float(np.mean(np.log(tp)))
    return np.concatenate([[np.log(target_shape) - mean_slope - mean_log_t], slopes])


def true_beta_for_regime(regime, rng, nu=DEFAULT_NU, slopes=DEFAULT_SLOPES,
                         t_range=T_RANGE, sparse_shape=None):
    """beta with the intercept set so the population geometric-mean count is
    the regime's target, or, for the sparse regime, the geometric-mean shape is
    sparse_shape."""
    if regime == "sparse":
        if sparse_shape is None:
            raise ValueError("the sparse regime needs sparse_shape (the paper uses 5)")
        return beta_for_shape(sparse_shape, rng, slopes, t_range)
    target_mu = REGIME_TARGET_MU[regime]
    Xp, tp = make_synthetic_design(200_000, rng, t_range)
    mean_slope = float(np.mean(Xp[:, 1:] @ slopes))
    mean_log_t = float(np.mean(np.log(tp)))
    # log mu = beta0 + slope + log t - log nu ; want exp(E[log mu]) = target_mu
    beta0 = np.log(target_mu) - mean_slope - mean_log_t + np.log(nu)
    return np.concatenate([[beta0], slopes])


# --------------------------------------------------------------------------- #
# One replication
# --------------------------------------------------------------------------- #
def _poisson_glm_mu(X, Y, t, iters=30):
    """Fitted means exp(X_i' b) t_i of a Poisson regression with offset log t (IRLS)."""
    offset = np.log(t)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.minimum(X @ b + offset, 30.0)
        mu = np.exp(eta)
        W = mu
        z = (eta - offset) + (Y - mu) / np.maximum(mu, 1e-8)
        XtW = X.T * W
        try:
            b_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(b_new - b)) < 1e-7:
            b = b_new
            break
        b = b_new
    return np.exp(np.minimum(X @ b + offset, 30.0))


def _moment_nu(Y, t, X=None):
    """Moment estimator of nu from Var(Y_i) = mu_i (1 + 1/nu): nu = 1/(D - 1),
    with D the Pearson dispersion about the Poisson-regression fit (or, without
    X, about the pooled-rate mean)."""
    if X is None:
        mu, dof = float(np.mean(Y / t)) * t, len(Y)
    else:
        mu, dof = _poisson_glm_mu(X, Y, t), max(len(Y) - X.shape[1], 1)
    D = float(np.sum((Y - mu) ** 2 / np.maximum(mu, 1e-12)) / dof)
    return np.inf if D <= 1.0 else 1.0 / (D - 1.0)


def _shape_lcb_factor(Y, t, X, B=200, seed=0):
    """exp(-1.645 s), s the bootstrap (B resamples) standard error of the log
    estimated shape."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    v = np.empty(B)
    for b in range(B):
        k = rng.integers(0, n, n)
        nb = _moment_nu(Y[k], t[k], X[k] if X is not None else None)
        v[b] = np.median(float(np.mean(Y[k] / t[k])) * t[k]) * nb
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(-1.645 * np.std(np.log(v)))) if len(v) > 20 else 1.0


def _norm_mask(partition, Y, alpha_t, X, t, threshold, nu=DEFAULT_NU, kappa=10.0,
               min_cont=0):
    """Normal-assignment mask of a rule (_raw_norm_mask), with the rank guard:
    fewer than min_cont Normal-assigned observations makes the fit entirely NB.
    Applied to every rule alike, so the rules are compared on the same data."""
    mask = _raw_norm_mask(partition, Y, alpha_t, X, t, threshold, nu, kappa)
    n_c = int(mask.sum())
    if 0 < n_c < min_cont:
        return np.zeros(len(Y), bool)       # too thin to be identified: all NB
    return mask


def _raw_norm_mask(partition, Y, alpha_t, X, t, threshold, nu=DEFAULT_NU,
                   kappa=10.0):
    """True where an observation is assigned to the Normal component.

      realized         Y_i >= threshold (the observation's own count)
      plugin_rate      pooled rate x exposure >= threshold (the mean rule)
      expected_shape   true shape alpha_t = exp(X_i' beta) t_i >= kappa (the oracle)
      plugin_shape     estimated shape (pooled mean x moment nu) >= kappa
      plugin_shape_lcb the same on its bootstrap lower bound
    """
    if partition == "realized":
        return Y >= threshold
    if partition == "plugin_rate":
        return float(np.mean(Y / t)) * t >= threshold
    if partition == "expected_shape":
        return alpha_t >= kappa
    if partition in ("plugin_shape", "plugin_shape_lcb"):
        nu_hat = _moment_nu(Y, t, X)
        phi = (float(np.mean(Y / t)) * t) * nu_hat
        if partition == "plugin_shape_lcb" and np.isfinite(nu_hat):
            phi = phi * _shape_lcb_factor(Y, t, X)
        return phi >= kappa
    raise ValueError(f"unknown partition: {partition}")


def run_one(rep, regime, n, exe_gfd, exe_bayes, exe_nb,
            beta_true, threshold, chains, warmup, sampling,
            adapt_delta, seed, partitions, methods, nu=DEFAULT_NU,
            t_range=T_RANGE, kappa=10.0, stan_seed=None):
    """One dataset, fit under every rule in `partitions` and every method in
    `methods`. `seed` generates the data and seeds Stan unless `stan_seed` is
    given (sim_retry_rhat.py refits the same data with fresh chains)."""
    import cmdstanpy
    cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
    from cmdstanpy import CmdStanModel
    rng = np.random.default_rng(seed)
    X, t = make_synthetic_design(n, rng, t_range)
    eta = X @ beta_true
    alpha_t = np.exp(eta) * t
    Y = rng.negative_binomial(alpha_t, nu / (1.0 + nu))
    params = list(zip(COVARIATE_NAMES, beta_true)) + [("nu", nu)]
    exes = {"gfd": exe_gfd, "bayes_flat": exe_bayes}

    def fit_and_cover(exe, method, data, partition, n_norm, n_nb, fit_model):
        model = CmdStanModel(exe_file=exe)
        try:
            fit = model.sample(
                data=data, chains=chains, parallel_chains=1,
                iter_warmup=warmup, iter_sampling=sampling,
                adapt_delta=adapt_delta, max_treedepth=12,
                seed=int(seed if stan_seed is None else stan_seed) % (2**31 - 1),
                show_progress=False, show_console=False,
            )
        except Exception as e:  # noqa: BLE001
            return [dict(regime=regime, n=n, rep=rep, method=method, partition=partition,
                         param="__fit_error__", n_norm=n_norm, n_nb=n_nb,
                         fit_model=fit_model, error=str(e)[:200])]
        beta_draws = fit.stan_variable("beta")
        nu_draws = fit.stan_variable("nu")
        rhat_max = float(fit.summary()["R_hat"].max())
        dbp = {COVARIATE_NAMES[k]: beta_draws[:, k] for k in range(len(COVARIATE_NAMES))}
        dbp["nu"] = nu_draws
        out = []
        for pname, ptrue in params:
            d = dbp[pname]
            lo95, hi95 = np.percentile(d, [2.5, 97.5])
            lo90, hi90 = np.percentile(d, [5.0, 95.0])
            out.append(dict(
                regime=regime, n=n, rep=rep, method=method, partition=partition,
                nu_true=nu, param=pname, true=ptrue,
                cov95=int(lo95 <= ptrue <= hi95),
                cov90=int(lo90 <= ptrue <= hi90),
                width95=hi95 - lo95, point=float(np.median(d)),
                u_rank=float(np.mean(d <= ptrue)),
                n_norm=n_norm, n_nb=n_nb, rhat_max=rhat_max, fit_model=fit_model
            ))
        return out

    records = []
    for partition in partitions:
        nm = _norm_mask(partition, Y, alpha_t, X, t, threshold, nu, kappa,
                        X.shape[1] + 1)
        nbm = ~nm
        n_norm, n_nb = int(nm.sum()), int(nbm.sum())
        data_hybrid = dict(r=X.shape[1], n_norm=n_norm, X_norm=X[nm].tolist(),
                           t_norm=t[nm].tolist(), Y_norm=Y[nm].astype(float).tolist(),
                           n_nb=n_nb, X_nb=X[nbm].tolist(), t_nb=t[nbm].tolist(),
                           Y_nb=Y[nbm].astype(int).tolist())
        data_nb = dict(n=n, r=X.shape[1], X=X.tolist(), t=t.tolist(),
                       Y=Y.astype(int).tolist())
        for method in methods:
            if method == "gfd" and n_norm == 0:
                # entirely NB: the NB likelihood alone, replaced downstream by
                # the discrete GFD
                records += fit_and_cover(exe_nb, method, data_nb, partition,
                                         n_norm, n_nb, "discrete_nb")
            else:
                records += fit_and_cover(exes[method], method, data_hybrid, partition,
                                         n_norm, n_nb,
                                         "hybrid" if method == "gfd" else "bayes_flat")
    return records


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=int, default=1000, help="replications per (regime, n)")
    ap.add_argument("--sparse-shape", type=float, default=5.0,
                    help="geometric-mean NB shape of the sparse regime")
    ap.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    ap.add_argument("--n_grid", nargs="+", type=int, default=DEFAULT_N_GRID)
    ap.add_argument("--threshold", type=float, default=500.0)
    ap.add_argument("--nu", type=float, default=DEFAULT_NU, help="true dispersion")
    ap.add_argument("--nu-range", type=str, default=None,
                    help="lo,hi: draw nu log-uniformly per replication (Section 5.4)")
    ap.add_argument("--omega", "--kappa", dest="kappa", type=float, default=10.0,
                    help="shape threshold omega of the shape rules")
    ap.add_argument("--partition", nargs="+", default=["plugin_rate"],
                    choices=["realized", "plugin_rate", "expected_shape", "plugin_shape",
                             "plugin_shape_lcb"],
                    help="partition rules, all fit to the same data")
    ap.add_argument("--methods", nargs="+", default=["gfd", "bayes_flat"],
                    choices=["gfd", "bayes_flat"])
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--adapt_delta", type=float, default=0.9)
    ap.add_argument("--n_jobs", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "sim_coverage_results.csv"))
    ap.add_argument("--quick", action="store_true", help="test run: R = 20, n in {50, 100}")
    args = ap.parse_args()
    slopes, t_range = DEFAULT_SLOPES, T_RANGE
    if args.quick:
        args.R, args.n_grid = 20, [50, 100]

    fits_per_rep = len(args.partition) * len(args.methods)
    total_fits = len(args.regimes) * len(args.n_grid) * args.R * fits_per_rep
    print(f"[plan] {len(args.regimes)} regimes x {len(args.n_grid)} n x R={args.R} "
          f"x {fits_per_rep} fits/rep  =>  {total_fits} fits, n_jobs={args.n_jobs}")
    print(f"[plan] partitions={args.partition}  methods={args.methods}")

    from joblib import Parallel, delayed
    import cmdstanpy
    from cmdstanpy import CmdStanModel
    cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
    exe_gfd = CmdStanModel(stan_file=STAN_FILE).exe_file
    exe_bayes = CmdStanModel(stan_file=STAN_FILE_FLAT).exe_file
    exe_nb = CmdStanModel(stan_file=STAN_FILE_NB).exe_file

    pilot_rng = np.random.default_rng(args.seed)
    ss = np.random.SeedSequence(args.seed)

    if os.path.exists(args.out):
        os.rename(args.out, args.out + ".bak")
        print(f"[out] existing results moved to {args.out}.bak")

    nu_range = None
    if args.nu_range:
        lo, hi = (float(v) for v in args.nu_range.split(","))
        nu_range = (lo, hi)

    first_write = True
    out_columns = None   # fixed by the first block, so every appended block matches
    for regime in args.regimes:
        if nu_range is None:
            beta_true = true_beta_for_regime(regime, pilot_rng, args.nu, slopes, t_range,
                                             sparse_shape=args.sparse_shape)
            print(f"\n=== regime={regime}  beta_true={np.round(beta_true,3)}  nu={args.nu} "
                  f"(NB shape at threshold = {args.threshold * args.nu:.1f}) ===")
        else:
            print(f"\n=== regime={regime}  nu ~ log-uniform{nu_range} per replication "
                  f"(shape at count threshold spans "
                  f"{args.threshold*nu_range[0]:.1f}-{args.threshold*nu_range[1]:.1f}; "
                  f"shape rules use kappa={args.kappa:g}) ===")
        for n in args.n_grid:
            child_seeds = ss.spawn(args.R)
            nu_rng = np.random.default_rng([args.seed, zlib.crc32(regime.encode()), n])
            if nu_range is None:
                nus = [args.nu] * args.R
                betas = [beta_true] * args.R
            else:
                nus = list(np.exp(nu_rng.uniform(np.log(nu_range[0]),
                                                 np.log(nu_range[1]), args.R)))
                betas = [true_beta_for_regime(regime,
                                              np.random.default_rng(1000 + i),
                                              nus[i], slopes, t_range,
                                              sparse_shape=args.sparse_shape)
                         for i in range(args.R)]
            tasks = (
                delayed(run_one)(
                    rep, regime, n, exe_gfd, exe_bayes, exe_nb, betas[rep],
                    args.threshold, args.chains, args.warmup, args.sampling,
                    args.adapt_delta, int(child_seeds[rep].generate_state(1)[0]),
                    args.partition, args.methods, nus[rep], t_range, args.kappa,
                )
                for rep in range(args.R)
            )
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(tasks)

            rows = [r for sub in results for r in sub]
            df = pd.DataFrame(rows)
            n_err = int((df.get("param") == "__fit_error__").sum()) if "param" in df else 0
            df = df[df.param != "__fit_error__"] if "param" in df else df

            if out_columns is None:
                out_columns = list(df.columns)
            else:
                extra = [c for c in df.columns if c not in out_columns]
                if extra:
                    print(f"  [warn] dropping columns absent from the header: {extra}")
                df = df.reindex(columns=out_columns)

            df.to_csv(args.out, mode="a", header=first_write, index=False)
            first_write = False

            ok = df[df.param.isin(COVARIATE_NAMES + ["nu"]) & (df.method == "gfd")]
            print(f"  n={n:<4d}  ({n_err} fit errors)   GFD 95% coverage by partition:")
            parts = args.partition
            fn = {pt: ok[ok.partition == pt].drop_duplicates("rep")["n_norm"].mean() / n
                  for pt in parts}
            print(f"     {'param':10s} " + " ".join(f"{pt:>13s}" for pt in parts))
            piv = ok.pivot_table(index="param", columns="partition", values="cov95",
                                 aggfunc="mean")
            for p in COVARIATE_NAMES + ["nu"]:
                cells = " ".join(f"{piv.loc[p, pt]:>13.3f}"
                                 if (p in piv.index and pt in piv.columns) else f"{'-':>13s}"
                                 for pt in parts)
                print(f"     {p:10s} {cells}")
            print(f"     {'%Normal':10s} " + " ".join(f"{fn[pt]:>12.0%} " for pt in parts))

    print(f"\n[done] results -> {args.out}")


if __name__ == "__main__":
    main()
