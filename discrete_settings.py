"""Settings of the discrete-GFD runs (Supplement, Section 8).

  cap           bounds (lambda_lo, lambda_hi) on log nu; part of the parameter
                set Theta, so they must lie far outside the fiducial mass.
  scan spacing  spacing h in log nu of the scan that locates the pieces of I(u)
                (discrete_gfi.feasible_pieces). A piece away from the current
                log nu is found only if it is wider than h; every run reports
                the narrowest such piece it found, so the choice can be checked.
"""

# Small-n validation (validate_small_n.py).
VALIDATION_CAP = (-6.0, 6.0)
VALIDATION_SCAN_SPACING = 0.02
REFERENCE_SCAN_SPACING = 0.005    # the exact reference has no anchor, so it scans finer

# Application (Section 6): 26 entirely discrete bins at n = 216.
APP_CAP = (-10.0, 0.0)
APP_SCAN_SPACING = 0.25

# Simulation study. The log-nu fiducial distributions of every cell lie more
# than eight posterior sd inside the cap, and in a pilot of eight replications
# at n = 50 and 500 spacings 0.05, 0.1 and 0.25 gave identical chains.
SIM_CAP = (-10.0, 0.0)
SIM_SCAN_SPACING = 0.25
