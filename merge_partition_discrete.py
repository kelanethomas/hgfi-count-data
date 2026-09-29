"""Put the discrete-GFD fits into the partition-rule sweep (Table 2).

In the Stan sweep (sim_coverage.py --nu-range), a fit that assigns nothing to
the Normal component is the NB-only model, i.e. the flat-prior posterior.
sim_partition_discrete.py fits the discrete GFD to those same datasets; this
replaces each such `gfd` row with the discrete fit, leaving the hybrid fits and
the `bayes_flat` rows unchanged. A dataset's discrete fit is recorded against
every rule that assigned nothing on it (the `rules` column), so each rule is
still evaluated on identical data. All discrete fits are kept.

    python merge_partition_discrete.py \
        --hb-sweep sim_partition_shape_all_merged.csv \
        --sparse-sweep sim_partition_sparse_shape5_merged.csv \
        --discrete 'part_disc_*.csv' \
        --out sim_partition_discrete_current_merged.csv
"""
import argparse
import glob
import sys

import numpy as np
import pandas as pd

from discrete_to_sweep import PARAM_MAP, nu_scale

KEY = ["regime", "n", "rep"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hb-sweep", required=True, help="Stan sweep; its heavy and borderline rows")
    ap.add_argument("--sparse-sweep", required=True, help="Stan sweep of the sparse regime")
    ap.add_argument("--discrete", nargs="+", required=True, help="sim_partition_discrete.py results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    hb = pd.read_csv(a.hb_sweep, low_memory=False)
    sw = pd.concat([hb[hb.regime != "sparse"], pd.read_csv(a.sparse_sweep, low_memory=False)],
                   ignore_index=True)
    if sw.duplicated(KEY + ["partition", "method", "param"]).any():
        sys.exit("duplicate rows in the sweep")

    files = sorted({p for pat in a.discrete for p in glob.glob(pat)})
    disc = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "refused" in disc and disc.refused.notna().any():
        sys.exit(f"{int(disc.refused.notna().sum())} refused rows; refit them before merging")
    if disc.duplicated(KEY + ["param"]).any():
        sys.exit("a dataset was fit twice across the discrete files")
    lnu = disc[disc.param == "log_nu"].set_index(KEY).gfd_ess
    ratio = lnu / disc[disc.param != "log_nu"].groupby(KEY).gfd_ess.mean()

    # one row per rule that assigned nothing on the dataset
    disc = disc.assign(partition=disc.rules.str.split(";")).explode("partition")
    disc = disc.reset_index(drop=True)
    true_v, width_v, point_v = nu_scale(disc)

    target = sw[(sw.method == "gfd") & (sw.n_norm == 0)]
    tkeys = set(map(tuple, target[KEY + ["partition"]].drop_duplicates().to_numpy()))
    dkeys = set(map(tuple, disc[KEY + ["partition"]].drop_duplicates().to_numpy()))
    if dkeys != tkeys:
        sys.exit(f"mismatch: {len(tkeys - dkeys)} NB-only fits without a discrete fit, "
                 f"{len(dkeys - tkeys)} discrete fits without an NB-only fit")

    rows = pd.DataFrame({
        "regime": disc.regime, "n": disc.n, "rep": disc.rep, "method": "gfd",
        "partition": disc.partition, "nu_true": disc.nu_true,
        "param": disc.param.map(PARAM_MAP), "true": true_v, "cov95": disc.gfd_cov,
        "cov90": np.nan, "width95": width_v, "point": point_v, "u_rank": np.nan,
        "n_norm": 0, "n_nb": disc.n, "rhat_max": 1.0, "fit_model": "discrete_gfd"})
    replaced = (sw.method == "gfd") & (sw.n_norm == 0)
    out = pd.concat([sw[~replaced], rows[sw.columns.intersection(rows.columns)]],
                    ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"{int(replaced.sum()):,} NB-only gfd rows replaced by {len(rows):,} discrete rows "
          f"from {len(ratio):,} datasets ({int((ratio < 0.8).sum())} with ESS(log nu) below "
          f"0.8 x ESS(beta), all kept) -> {a.out}")


if __name__ == "__main__":
    main()
