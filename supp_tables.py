"""Supplement Tables 1 and 2: coverage of the 95% fiducial intervals by regime,
parameter and n at nu = 0.05, and under assignment by the realized count
(borderline regime). Stan fits not converged after the refit are excluded.

    python supp_tables.py
"""
import numpy as np
import pandas as pd

PARAMS = ["Intercept", "X_cont", "X_grpA", "X_grpB", "nu"]


def half_up(t):
    """Three decimals, exact ties (R = 1000 or 2000 gives multiples of 0.0005)
    rounded up."""
    return t.map(lambda v: f"{np.floor(v * 1000 + 0.5 + 1e-9) / 1000:.3f}")


def coverage(path, **where):
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.method == "gfd") & (d.param != "__fit_error__")]
    for k, v in where.items():
        d = d[d[k] == v]
    first = d.groupby(["regime", "n", "rep"]).rhat_max.transform("first")
    return d[(d.fit_model == "discrete_gfd") | (first <= 1.05)]


d = coverage("sim_coverage_main_discrete_final.csv")
print("Supplement Table 1: coverage at nu = 0.05")
for rg in ["sparse", "borderline", "heavy"]:
    t = d[d.regime == rg].pivot_table(index="param", columns="n", values="cov95").loc[PARAMS]
    t.loc["Average"] = t.mean()
    print(f"\n{rg}\n{half_up(t).to_string()}")

d = coverage("sim_partition_complete.csv", partition="realized")
print("\nSupplement Table 2: assignment by the realized count (borderline)")
print(half_up(d.pivot_table(index="param", columns="n", values="cov95").loc[PARAMS]).to_string())
