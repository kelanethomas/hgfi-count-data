"""Table 2: average coverage of the 95% fiducial intervals over the five
parameters, by regime, partition rule and n, with the range over n of the
average fraction of observations assigned to the Normal component, and the
counts of excluded fits (Section 5.1; Supplement, Section 7).

Input is the merged partition sweep (merge_partition_discrete.py): hybrid Stan
fits where a rule assigned something to the Normal component, discrete-GFD fits
where it assigned nothing. Hybrid fits not converged after the refit (R-hat
above 1.05, or none) are excluded; the effect of including them is printed.

    python table2_partition.py [sim_partition_discrete_current_merged.csv]
"""
import sys
import numpy as np
import pandas as pd

hu = lambda v: np.floor(v * 1000 + 0.5 + 1e-9) / 1000   # exact ties rounded up

RULES = [("expected_shape", "Oracle shape"), ("plugin_rate", "Mean, tau=500"),
         ("plugin_shape", "Shape, omega=10"), ("plugin_shape_lcb", "Lower-bound shape")]
REGIMES = ["sparse", "borderline", "heavy"]

f = sys.argv[1] if len(sys.argv) > 1 else "sim_partition_discrete_current_merged.csv"
d = pd.read_csv(f, low_memory=False)
d = d[(d.method == "gfd") & (d.param != "__fit_error__")].copy()
key = ["regime", "n", "rep", "partition"]
first = d.groupby(key)[["rhat_max", "n_norm"]].transform("first")
# not converged: R-hat above 1.05, or none (a chain that did not move)
bad = (first.n_norm > 0) & ~(first.rhat_max <= 1.05)
fits = d[bad].drop_duplicates(key)
nan = int(fits.rhat_max.isna().sum())
print(f"{f}: {d.drop_duplicates(key).shape[0]:,} fits; hybrid fits not converged after the "
      f"refit: {len(fits)} ({nan} with no R-hat) {fits.groupby(['regime', 'partition']).size().to_dict()}")
ok = d[~bad]
cov = ok.groupby(["regime", "partition", "n"]).cov95.mean()
cov_all = d.groupby(["regime", "partition", "n"]).cov95.mean()
# the assigned fraction is a property of the rule and the data: all fits
assigned = d.drop_duplicates(key).assign(frac=lambda s: s.n_norm / s.n) \
    .groupby(["regime", "partition", "n"]).frac.mean()
print(f"\n{'Regime':11} {'Rule':19} {'Normal-assigned':>15}   {'50':>6} {'100':>6} {'200':>6} {'500':>6}")
for rg in REGIMES:
    for p, lab in RULES:
        if (rg, p) not in cov.droplevel("n").index:
            continue
        r = assigned.loc[(rg, p)]
        lo, hi = 100 * r.min(), 100 * r.max()
        fmt = (lambda v: f"{v:.1f}".rstrip("0").rstrip(".")) if hi < 1 else (lambda v: f"{v:.0f}")
        rr = f"{fmt(lo)}%" if fmt(lo) == fmt(hi) else f"{fmt(lo)}-{fmt(hi)}%"
        print(f"{rg:11} {lab:19} {rr:>15}   " + " ".join(f"{hu(cov.loc[(rg, p, n)]):6.3f}" for n in (50, 100, 200, 500)))
diff = (cov_all - cov)
print(f"\nincluding the non-converged fits would change cells by at most {diff.abs().max():.4f}; "
      f"cells moving by 0.002 or more:")
print(diff[diff.abs() >= 0.002].round(4).to_string())
