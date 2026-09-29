# Code for "Generalized Fiducial Inference for Hybrid Discrete–Continuous Count Data"

Thomas and Hannig. Code and results for the simulation study (Section 5 of the
paper; Supplement, Sections 7–8) and for the discrete generalized fiducial
distribution (Section 2.1; Supplement, Section 8). The application (Section 6)
uses GPS data collected under IRB 25-1234, whose data-sharing agreement does not
permit redistribution, so neither those data nor the application code are
included.

## Setup

Python 3.14 with the packages in `requirements.txt`, and CmdStan 2.38 for the
Stan models (`python -m cmdstanpy.install_cmdstan --version 2.38.0`). All
commands are run from this folder.

## Files

**Samplers**

| File | Contents |
|---|---|
| `gfd_theta_mh.py` | MH sampler for the discrete GFD (Supplement, Algorithm 1) |
| `discrete_gfi.py` | Single-site Gibbs sampler; the feasible log-ν set and linear programs shared by both samplers |
| `discrete_settings.py` | Cap on log ν and scan spacing for each use |
| `model_single_bin_hybrid.stan` | Hybrid fiducial density: likelihood times Jacobian (Theorem 2) |
| `model_single_bin_hybrid_flat.stan` | Flat-prior posterior, the comparator |
| `model_bin_nb.stan` | NB likelihood alone, for fits with nothing assigned to the Normal component |

**Validation** (Supplement, Section 8.4)

| File | Contents |
|---|---|
| `exact_ref2.py` | Exact rejection sampler from the definition of the discrete GFD |
| `validate_small_n.py` | Both samplers against the exact reference (Supplement, Table 3) |
| `check_failed_proposals.py` | 50-digit recomputation of the proposals the MH sampler could not evaluate |

**Simulation study**

| File | Contents |
|---|---|
| `sim_coverage.py` | Stan sweep: data generation, partition rules, hybrid and flat-prior fits |
| `sim_retry_rhat.py` | Refits of non-converged Stan fits |
| `sim_fill_rank_guard.py` | Discrete GFD for datasets on which the mean rule assigns at most K observations to the Normal component |
| `sim_discrete_coverage.py` | Sparse regime: discrete GFD and its flat-prior comparator |
| `discrete_to_sweep.py` | Sparse results into the sweep format, with the heavy and borderline rows |
| `sim_partition_discrete.py`, `merge_partition_discrete.py` | Discrete GFD for the entirely-NB fits of the partition-rule comparison |
| `table1_dispersion.py`, `table2_partition.py` | Tables 1 and 2 and the numbers of Sections 5.3–5.4 |
| `supp_tables.py` | Supplement Tables 1 and 2 |
| `sim_bvm_figure.py`, `plot_calibration.py`, `plot_interval_lengths.py` | Figures 1, 2 and 3 |

**Results** (the files the tables and figures are built from)

| File | Contents |
|---|---|
| `sim_coverage_main_discrete_final.csv` | Coverage study at ν = 0.05, all regimes (Section 5.2; Supplement, Table 1) |
| `sparse_shape5_n{50,100,200,500}_final.csv` | Sparse regime, discrete GFD and flat-prior comparator per replication (Supplement, Section 8.5) |
| `sim_coverage_nu0.005_merged_filled.csv`, `sim_coverage_nu0.02_merged_filled.csv` | Dispersion sensitivity (Table 1) |
| `sim_partition_discrete_current_merged.csv` | Partition-rule comparison (Table 2) |
| `sim_partition_complete.csv` | Assignment by the realized count (Supplement, Table 2) |
| `sim_bvm_figure_draws.npz` | Draws behind Figure 1 |

One row per replication, method, partition rule and parameter; `cov95` is
coverage of the 95% interval, `width95` its length, `n_norm` the number of
observations assigned to the Normal component, `rhat_max` the largest R-hat
(none for discrete fits). Coverages are proportions of 1,000 or 2,000
replications, so many are exact ties at three decimals; the table scripts round
ties up.

## Tables and figures from the results

```bash
python table1_dispersion.py 0.005:sim_coverage_nu0.005_merged_filled.csv:1000 \
    0.02:sim_coverage_nu0.02_merged_filled.csv:1000 0.05:sim_coverage_main_discrete_final.csv:2000
python table2_partition.py
python supp_tables.py
python sim_bvm_figure.py --replot
python plot_calibration.py
python plot_interval_lengths.py      # figures are written to figures/
```

## Rerunning the simulation study

Every dataset is determined by its seed, so each step below regenerates the
datasets of the step before it and checks them against the recorded results.
Costs are CPU core-hours, approximately.

```bash
# Coverage study at nu = 0.05 (Section 5.2): heavy and borderline      ~2,000
python sim_coverage.py --regimes heavy borderline --R 2000 --out sim_coverage_full.csv
python sim_retry_rhat.py --in sim_coverage_full.csv --nu 0.05 --R 2000

# Sparse regime, discrete GFD (one run per n)                          ~9,000
python sim_discrete_coverage.py --R 2000 --n 50 --nu 0.05 --shape 5 --out sparse_shape5_n50_final.csv
python discrete_to_sweep.py --discrete sparse_shape5_n*_final.csv \
    --hb sim_coverage_full_merged.csv --out sim_coverage_main_discrete_final.csv

# Dispersion sensitivity (Section 5.3), nu = 0.005 and 0.02            ~2,000
python sim_coverage.py --regimes heavy borderline --nu 0.005 --R 1000 --out sim_coverage_nu0.005.csv
python sim_retry_rhat.py --in sim_coverage_nu0.005.csv --nu 0.005 --R 1000 --regimes heavy borderline
python sim_fill_rank_guard.py --in sim_coverage_nu0.005_merged.csv --nu 0.005 --R 1000 --regimes heavy borderline
# and the same three lines with 0.02 in place of 0.005

# Partition rules (Section 5.4)                                        ~6,000
python sim_coverage.py --regimes heavy borderline --nu-range 0.002,0.07 --methods gfd \
    --partition expected_shape plugin_rate plugin_shape plugin_shape_lcb \
    --R 1000 --out sim_partition_shape_all.csv
python sim_coverage.py --regimes sparse --sparse-shape 5 --nu-range 0.002,0.07 --methods gfd \
    --partition expected_shape plugin_rate plugin_shape plugin_shape_lcb \
    --R 1000 --out sim_partition_sparse_shape5.csv
python sim_retry_rhat.py --in sim_partition_shape_all.csv
python sim_retry_rhat.py --in sim_partition_sparse_shape5.csv --regimes sparse
python sim_partition_discrete.py --inp sim_partition_shape_all_merged.csv \
    --spawn-regimes heavy borderline --fit-regimes heavy borderline --out part_disc_hb.csv
python sim_partition_discrete.py --inp sim_partition_sparse_shape5_merged.csv \
    --spawn-regimes sparse --fit-regimes sparse --sparse-shape 5 --out part_disc_sp.csv
python merge_partition_discrete.py --hb-sweep sim_partition_shape_all_merged.csv \
    --sparse-sweep sim_partition_sparse_shape5_merged.csv --discrete 'part_disc_*.csv' \
    --out sim_partition_discrete_current_merged.csv

# Assignment by the realized count (Supplement, Table 2)               ~200
python sim_coverage.py --regimes borderline --partition realized --methods gfd --R 2000 \
    --out sim_partition_complete.csv
python sim_retry_rhat.py --in sim_partition_complete.csv --nu 0.05 --R 2000 --regimes borderline

# Figure 1                                                             <1
python sim_bvm_figure.py

# Validation of the samplers (Supplement, Section 8.4)                 ~50
python validate_small_n.py --dataset n4 --ref 4000 --ref2 4000 --gibbs 20000 --chains 16 --mh 16000 --n_jobs 16
python check_failed_proposals.py --chains 16 --steps 16000 --below 100 --n_jobs 16
```

Every script takes `--n_jobs` for parallel replications, and the long ones take
`--reps` or `--rep-lo/--rep-hi` to split a run; results do not depend on either.
