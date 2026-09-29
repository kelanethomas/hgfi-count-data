"""Single-site Gibbs sampler for the discrete generalized fiducial distribution
of one entirely-NB component with covariates (Supplement, Section 8), and the
geometry it shares with the MH sampler of gfd_theta_mh.py.

The sampler extends Hannig, Iyer and Wang (2007) to NB regression. It is the
efficient choice at small n and the second reference in the validation
(validate_small_n.py); at n = 216 its single-site updates do not mix. The shared
utilities are nb_cdf, eta_bounds, CenterLP, rescaled_center and feasible_pieces
(the feasible log-nu set), r_solve (used by exact_ref2.py) and nb_mle.

Model for one component, observation i:
    Y_i ~ NB(r_i, p),  r_i = exp(X_i' beta) t_i,  p = nu / (1 + nu),
    F(y; r, p) = I_p(r, y + 1),  F(-1; r, p) = 0.
theta = (beta, nu) reproduces y_i from U_i when F(y_i - 1; theta) < U_i <= F(y_i; theta).
F is decreasing in r, so at fixed nu each observation confines X_i' beta to an
interval and the consistent beta form a convex polytope P_U(log nu); feasibility
and extreme values of X_j' beta are linear programs (HiGHS, warm-started).

Gibbs sweep:
  1. For each j, draw U_j uniformly on the values that keep Q(y, U) nonempty
     given the other U: the union, over every piece of the leave-one-out
     feasible log-nu set, of (smallest F(y_j - 1), largest F(y_j)).
  2. Select theta: log nu uniform on I(U), then beta uniform on P_U(log nu)
     by hit-and-run.

beta is confined to |beta_k| < beta_box and log nu to log_nu_bounds (the set
Theta of the Supplement). The box binds only when the rows with y > 0 fail to
span R^K (`unbounded_by_design`).
"""
import numpy as np
import highspy
from scipy.optimize import minimize, brentq
from scipy.special import betainc, gammaln

INF = highspy.kHighsInf
_OPT = highspy.HighsModelStatus.kOptimal
_INFEAS = highspy.HighsModelStatus.kInfeasible

LP_FAILURES = {"count": 0}
# How often the LPs were re-solved at the polytope's own scale (rescaled_center,
# _slack_value), and how often the first Chebyshev center was not inside by half
# its own radius, the case the re-solve exists for. Per process.
RESCALED = {"center_resolved": 0, "center_corrected": 0, "slack_resolved": 0,
            "center_cold_retry": 0}


# ----------------------------------------------------------------------------
# NB distribution function and its inverse in the shape r
# ----------------------------------------------------------------------------

def nb_cdf(y, r, p):
    """F(y; r, p) = I_p(r, y + 1) for y >= 0, and 0 for y < 0.

    Limits in r: F -> 1 as r -> 0 and F -> 0 as r -> inf (for y >= 0).
    """
    y = np.asarray(y, dtype=float)
    r = np.asarray(r, dtype=float)
    with np.errstate(invalid="ignore"):
        f = betainc(np.where(np.isfinite(r) & (r > 0), r, 1.0),
                    np.maximum(y, 0.0) + 1.0, p)
    f = np.where(r <= 0, 1.0, f)
    f = np.where(np.isposinf(r), 0.0, f)
    return np.where(y >= 0, f, 0.0)


def r_solve(u, y, p, log_lo=-40.0, log_hi=40.0, iters=50, guess=None, halfwidth=3.0):
    """Solve F(y; R, p) = u for R > 0, vectorized; F is decreasing in R.

    Bisection on log R. Cold start: 50 iterations on an 80-wide bracket give
    ~7e-14 in log R, below betainc precision. With `guess` (previous log R,
    same shape as the broadcast result) the bracket is guess +- halfwidth and
    ~40 iterations suffice; elements whose warm bracket misses the root fall
    back to the cold bracket. Broadcasts, so p of shape (G, 1) against u, y of
    shape (n,) yields a (G, n) result.
    """
    u, y, p = np.broadcast_arrays(np.asarray(u, float), np.asarray(y, float),
                                  np.asarray(p, float))
    b = y + 1.0
    if guess is None:
        lo = np.full(u.shape, log_lo)
        hi = np.full(u.shape, log_hi)
    else:
        g = np.broadcast_to(np.asarray(guess, float), u.shape)
        lo, hi = g - halfwidth, g + halfwidth
        ok = (betainc(np.exp(lo), b, p) > u) & (betainc(np.exp(hi), b, p) <= u)
        lo = np.where(ok, lo, log_lo)
        hi = np.where(ok, hi, log_hi)
        iters = max(iters, 50) if not ok.all() else int(np.ceil(np.log2(2 * halfwidth / 1e-12)))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = betainc(np.exp(mid), b, p) > u   # F above u -> R must be larger
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return np.exp(0.5 * (lo + hi))


def eta_bounds(u, y, t, nu, guess=None, return_logR=False):
    """Bounds (lower, upper] on X' beta implied by each (u_i, y_i).

    nu may be a scalar (returns shape (n,)) or an array of shape (G,) (returns
    shape (G, n)). Lower is -inf where y = 0. `guess` is an optional pair
    (logR_lower, logR_upper) from a previous call at nearby nu / U, used to
    warm-start the root finder; with return_logR=True that pair is returned
    as a third value for the next call.
    """
    u = np.asarray(u, float)
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    nu = np.asarray(nu, float)
    p = nu / (1.0 + nu)
    if p.ndim == 1:
        p = p[:, None]
    g_lo, g_hi = (None, None) if guess is None else guess
    R_hi = r_solve(u, y, p, guess=g_hi)
    logR_hi = np.log(R_hi)
    upper = logR_hi - np.log(t)
    lower = np.full_like(upper, -np.inf)
    logR_lo = np.full_like(upper, np.nan)
    pos = y > 0
    if np.any(pos):
        gl = None if g_lo is None else g_lo[..., pos]
        R_lo = r_solve(u[pos], y[pos] - 1, p, guess=gl)
        logR_lo[..., pos] = np.log(R_lo)
        lower[..., pos] = logR_lo[..., pos] - np.log(t[pos])
    if return_logR:
        return lower, upper, (logR_lo, logR_hi)
    return lower, upper


# ----------------------------------------------------------------------------
# Persistent LP models
# ----------------------------------------------------------------------------

def _new_highs():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


class RangeLP:
    """min / max of c' beta over  lower <= X beta <= upper,  |beta| <= box.

    One HiGHS model with k columns and n rows; bounds and objective are changed
    in place between solves.
    """

    def __init__(self, X, box):
        X = np.asarray(X, float)
        self.n, self.k = X.shape
        self.h = _new_highs()
        self.h.addVars(self.k, np.full(self.k, -box), np.full(self.k, box))
        starts = np.arange(0, self.n * self.k, self.k, dtype=np.int32)
        idx = np.tile(np.arange(self.k, dtype=np.int32), self.n)
        self.h.addRows(self.n, np.full(self.n, -INF), np.full(self.n, INF),
                       self.n * self.k, starts, idx, X.ravel())
        self._cols = np.arange(self.k, dtype=np.int32)
        self._rows = np.arange(self.n, dtype=np.int32)
        self.lower = np.full(self.n, -INF)
        self.upper = np.full(self.n, INF)

    def set_bounds(self, lower, upper):
        self.lower = np.where(np.isfinite(lower), lower, -INF)
        self.upper = np.where(np.isfinite(upper), upper, INF)
        self.h.changeRowsBounds(self.n, self._rows, self.lower, self.upper)

    def _solve(self, c):
        self.h.changeColsCost(self.k, self._cols, np.asarray(c, float))
        self.h.run()
        st = self.h.getModelStatus()
        if st == _OPT:
            return self.h.getInfo().objective_function_value
        if st != _INFEAS:
            LP_FAILURES["count"] += 1
        return None

    def range(self, c, drop=None):
        """(min, max) of c' beta, or None if the polytope is empty."""
        if drop is not None:
            self.h.changeRowBounds(int(drop), -INF, INF)
        lo = self._solve(c)
        hi = None if lo is None else self._solve(-np.asarray(c, float))
        if drop is not None:
            self.h.changeRowBounds(int(drop), self.lower[drop], self.upper[drop])
        if lo is None or hi is None:
            return None
        return lo, -hi


class CenterLP:
    """Chebyshev center / uniform-slack LP for  lower <= X beta <= upper.

    Columns: beta (k, boxed) and s. Rows, for each observation i:
        X_i beta + a_i s <= upper_i           (row i)
       -X_i beta + a_i s <= -lower_i          (row n + i, inactive when y_i = 0)
    With a_i = ||X_i|| and s >= 0 maximized, s is the inscribed radius and
    beta the Chebyshev center. With a_i = -1 and s free minimized, s is the
    smallest uniform slack (negative iff the set has interior), which is
    continuous in nu and usable for locating a feasible nu.
    """

    def __init__(self, X, box, mode):
        X = np.asarray(X, float)
        self.X = X
        self.n, self.k = X.shape
        self.mode = mode
        self.h = _new_highs()
        if mode == "center":
            a = np.linalg.norm(X, axis=1)
            s_lo, s_hi, cost = 0.0, INF, -1.0
        else:
            a = -np.ones(self.n)
            s_lo, s_hi, cost = -INF, INF, 1.0
        self._col_lo, self._col_hi = np.full(self.k, -float(box)), np.full(self.k, float(box))
        self.h.addVars(self.k + 1,
                       np.concatenate([self._col_lo, [s_lo]]),
                       np.concatenate([self._col_hi, [s_hi]]))
        c = np.zeros(self.k + 1)
        c[-1] = cost
        self.h.changeColsCost(self.k + 1, np.arange(self.k + 1, dtype=np.int32), c)
        M = np.vstack([np.hstack([X, a[:, None]]), np.hstack([-X, a[:, None]])])
        m = 2 * self.n
        starts = np.arange(0, m * (self.k + 1), self.k + 1, dtype=np.int32)
        idx = np.tile(np.arange(self.k + 1, dtype=np.int32), m)
        self.h.addRows(m, np.full(m, -INF), np.full(m, INF),
                       m * (self.k + 1), starts, idx, M.ravel())
        self._rows = np.arange(m, dtype=np.int32)

    def set_col_bounds(self, lo=None, hi=None):
        """Bounds on the beta columns; the construction's +-box when omitted."""
        lo = self._col_lo if lo is None else lo
        hi = self._col_hi if hi is None else hi
        self.h.changeColsBounds(self.k, np.arange(self.k, dtype=np.int32),
                                np.asarray(lo, float), np.asarray(hi, float))

    def solution(self):
        """The beta columns of the last optimal solution."""
        return np.array(self.h.getSolution().col_value)[:self.k]

    def set_bounds(self, lower, upper):
        hi = np.concatenate([np.where(np.isfinite(upper), upper, INF),
                             np.where(np.isfinite(lower), -lower, INF)])
        self.h.changeRowsBounds(2 * self.n, self._rows, np.full(2 * self.n, -INF), hi)

    def solve(self, cold=False):
        """center mode: (beta, radius) or None.  slack mode: slack value (inf
        if the LP is infeasible, which cannot happen with s free). cold=True
        discards the previous basis first (see rescaled_center)."""
        if cold:
            self.h.clearSolver()
        self.h.run()
        st = self.h.getModelStatus()
        if st == _OPT:
            if self.mode == "center":
                x = np.array(self.h.getSolution().col_value)
                return x[:self.k], x[self.k]
            return self.h.getInfo().objective_function_value
        if st != _INFEAS:
            LP_FAILURES["count"] += 1
        return None if self.mode == "center" else np.inf


# ----------------------------------------------------------------------------
# Pieces of the feasible log-nu set (shared by both samplers)
# ----------------------------------------------------------------------------

SLACK_RESOLVE = 1e-5     # re-solve the slack LP at the set's own scale below this


def _slack_value(slack_lp, lower, upper, drop=None):
    """Uniform-slack LP value for the bounds (lower, upper], optionally
    without observation `drop`: negative iff the beta-set has interior.

    The LP's feasibility tolerance is absolute, on bounds of order 1, so near
    zero its sign is unreliable once the eta windows are narrow (large counts).
    A value within SLACK_RESOLVE of zero is therefore recomputed around the
    LP's own solution beta1 and in units of the median window width w: bounds
    (bound - X beta1) / w, beta columns shifted and scaled alike. The map is
    affine and scales every row by the same factor, so the sign is unchanged
    in exact arithmetic; the value is returned in the original units."""
    if drop is not None:
        lower, upper = lower.copy(), upper.copy()
        lower[drop], upper[drop] = -np.inf, np.inf
    slack_lp.set_bounds(lower, upper)
    v = slack_lp.solve()
    if not np.isfinite(v):
        return 1.0
    if abs(v) >= SLACK_RESOLVE:
        return v
    fin = np.isfinite(lower) & np.isfinite(upper)
    w = float(np.median(upper[fin] - lower[fin])) if fin.any() else 1.0
    if not w > 0:
        return v
    b1 = slack_lp.solution()
    Xb = slack_lp.X @ b1
    slack_lp.set_bounds((lower - Xb) / w, (upper - Xb) / w)
    slack_lp.set_col_bounds((slack_lp._col_lo - b1) / w, (slack_lp._col_hi - b1) / w)
    v2 = slack_lp.solve()
    slack_lp.set_col_bounds()
    RESCALED["slack_resolved"] += 1
    return v2 * w if np.isfinite(v2) else v


def rescaled_center(center_lp, lower, upper):
    """Chebyshev center (z, radius) of {lower <= R z <= upper} within the LP's
    column bounds, R being center_lp's rows, or None.

    Solved twice: once as given, then again in x = (z - z0) / r0 about the
    first answer (z0, r0), where the set has inscribed radius about 1. The LP's
    feasibility tolerance is absolute, so when the set is small (inscribed
    radius near 1e-6 at large counts) the first center is accurate only to a
    fraction of the radius and can lie outside a face; in the rescaled problem
    the tolerance is relative to the set. The column bounds are rescaled with
    it and restored afterwards. The map is affine, so the set, and everything
    computed from it, is unchanged."""
    center_lp.set_bounds(lower, upper)
    cc = center_lp.solve()
    if cc is None or cc[1] <= 0:
        # The LP object is reused across calls and warm-starts from the last
        # basis; very rarely that start ends in a non-optimal status on an
        # ordinary polytope. One cold solve settles it.
        RESCALED["center_cold_retry"] += 1
        cc = center_lp.solve(cold=True)
        if cc is None or cc[1] <= 0:
            return None
    z0, r0 = cc
    R = center_lp.X
    Rz = R @ z0
    nrm = np.linalg.norm(R, axis=1)
    with np.errstate(invalid="ignore"):
        margin = np.concatenate([(upper - Rz) / nrm, (Rz - lower) / nrm])
    if np.nanmin(margin[np.isfinite(margin)]) < 0.5 * r0:
        RESCALED["center_corrected"] += 1
    center_lp.set_bounds((lower - Rz) / r0, (upper - Rz) / r0)
    center_lp.set_col_bounds((center_lp._col_lo - z0) / r0, (center_lp._col_hi - z0) / r0)
    cc2 = center_lp.solve()
    if cc2 is None or cc2[1] <= 0:
        RESCALED["center_cold_retry"] += 1
        cc2 = center_lp.solve(cold=True)
    center_lp.set_col_bounds()
    RESCALED["center_resolved"] += 1
    if cc2 is None or cc2[1] <= 0:
        return None
    return z0 + r0 * cc2[0], r0 * cc2[1]


def feasible_pieces(U, y, t, slack_lp, scan, anchors=(), drop=None, bounds_fn=None):
    """Pieces (a, c) of {log nu in [scan[0], scan[-1]] : the beta-set is
    nonempty} for the uniforms U, without observation `drop` if given.

    The slack is evaluated at the scan points and at the anchors (log nu values
    known to be feasible); each maximal run of feasible points is one piece,
    and each end strictly inside the scan range is located by Brent's method on
    the slack, which is continuous in log nu. A piece holding an anchor is found
    however narrow; any other piece is found only if a scan point falls in it,
    and two pieces separated by a gap narrower than the spacing are read as one.
    `slack_lp` is a CenterLP in "slack" mode built on the same design and box.

    Returns the pieces and a dict with, per piece, whether it reaches an end of
    the scan range (`capped`) and whether it holds an anchor (`anchored`), plus
    the number of ends at which the scan and the re-solve disagreed.

    `bounds_fn`, if given, replaces eta_bounds: it takes an array of log-nu
    values and returns the (lower, upper) eta bounds, each of shape (S, n). It
    is used by check_failed_proposals.py to recompute the set in high precision.
    """
    if bounds_fn is None:
        bounds_fn = lambda lnus: eta_bounds(U, y, t, np.exp(lnus))
    lo_cap, hi_cap = float(scan[0]), float(scan[-1])
    extra = [float(x) for x in anchors if x is not None and lo_cap < x < hi_cap]
    pts = np.unique(np.concatenate([np.asarray(scan, float), extra]))
    lower, upper = bounds_fn(pts)                                       # (S, n)
    feas = np.array([_slack_value(slack_lp, lower[s], upper[s], drop) < 0
                     for s in range(len(pts))])
    info = dict(capped=[], anchored=[], edge_disagreements=0)

    def f(v):
        lo, hi = bounds_fn(np.array([v]))
        return _slack_value(slack_lp, lo[0], hi[0], drop)

    def edge(inside, outside):
        # The scan classified inside as feasible and outside as not; a re-solve
        # can flip a point sitting on the boundary to machine precision, and the
        # end is then that point. Counted so its effect can be checked.
        fi, fo = f(inside), f(outside)
        if fi >= 0:
            info["edge_disagreements"] += 1
            return inside
        if fo < 0:
            info["edge_disagreements"] += 1
            return outside
        a, b = (inside, outside) if inside < outside else (outside, inside)
        try:
            return brentq(f, a, b, xtol=1e-12, maxiter=200)
        except ValueError:
            info["edge_disagreements"] += 1
            return inside

    pieces, i, S = [], 0, len(pts)
    while i < S:
        if not feas[i]:
            i += 1
            continue
        k = i                                       # maximal run of feasible points
        while k + 1 < S and feas[k + 1]:
            k += 1
        a = pts[0] if i == 0 else edge(pts[i], pts[i - 1])
        c = pts[-1] if k == S - 1 else edge(pts[k], pts[k + 1])
        if c > a:
            pieces.append((float(a), float(c)))
            info["capped"].append(i == 0 or k == S - 1)
            info["anchored"].append(any(a <= x <= c for x in extra))
        i = k + 1
    return pieces, info


def hit_and_run(A, b, x0, steps, rng, L=None):
    """Approximately uniform point in {A beta <= b}, starting from interior x0.

    Directions are L z with z standard normal (L = None gives isotropic). Any
    direction distribution with full support leaves the uniform distribution
    invariant, but when the polytope is elongated (correlated covariates)
    isotropic directions mix very slowly; L should be a Cholesky factor of a
    covariance with roughly the polytope's shape.
    """
    x = x0.copy()
    for _ in range(steps):
        z = rng.standard_normal(len(x))
        d = z if L is None else L @ z
        d /= np.linalg.norm(d)
        ad = A @ d
        slack = b - A @ x
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = slack / ad
        pos, neg = ad > 1e-14, ad < -1e-14
        if not (np.any(pos) and np.any(neg)):
            continue
        lam_hi, lam_lo = np.min(ratio[pos]), np.max(ratio[neg])
        if lam_hi > lam_lo:
            x = x + rng.uniform(lam_lo, lam_hi) * d
    return x


def _halfspaces(X, lower, upper, box):
    """A beta <= b for lower <= X beta <= upper, |beta| <= box (hit-and-run)."""
    fin = np.isfinite(lower)
    k = X.shape[1]
    A = np.vstack([X, -X[fin], np.eye(k), -np.eye(k)])
    b = np.concatenate([upper, -lower[fin], np.full(2 * k, box)])
    return A, b


# ----------------------------------------------------------------------------
# Starting point
# ----------------------------------------------------------------------------

def nb_mle(X, y, t):
    """NB regression MLE on (beta, log nu), used only to initialize the chain."""
    def nll(par):
        beta, lnu = par[:-1], par[-1]
        nu = np.exp(lnu)
        r = np.exp(np.clip(X @ beta, -30, 30)) * t
        p = nu / (1 + nu)
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(p) + y * np.log1p(-p))
        return -np.sum(ll)
    beta0 = np.linalg.lstsq(X, np.log((y + 0.5) / t) + np.log(0.05), rcond=None)[0]
    res = minimize(nll, np.concatenate([beta0, [np.log(0.05)]]), method="BFGS")
    return res.x[:-1], float(np.exp(res.x[-1])), res


# ----------------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------------

class DiscreteGFI:
    """Gibbs sampler for the discrete GFD of one entirely-NB bin, continuous in nu.

    For a single draw of the uniforms at realistic n, the consistent set is very
    narrow in nu, much narrower than the spread of the fiducial distribution, so
    a fixed nu grid would condition on the grid meeting the set rather than on the
    set being nonempty. Nu is therefore handled continuously: feasible log-nu sets
    are located by a scan of the cap range plus root-finding on the uniform-slack
    LP, and grids are laid only over sets already located.

    Sweep: for each j, find every piece of the leave-one-out feasible log-nu set,
    compute on each piece the interval of U_j values the other observations
    allow, and draw U_j uniformly on the union of those intervals; then select
    theta from Q(y, U). The block comment above _slack gives the details.

    Selection V: log nu uniform on the whole feasible set I(U), then beta uniform
    on its polytope by hit-and-run.

    Resolution: the scan has spacing `scan_spacing` in log nu. A piece narrower
    than that is found only if it contains an anchor (a log nu known to be
    feasible), so the split-set counters are lower bounds; `min_piece_width` and
    `narrow_pieces` (for I(U)) and their `_loo` versions (for the leave-one-out
    sets) report how close the pieces found came to the resolution. Those are
    dominated by the chain's own piece, which the anchor always finds; the
    check on the spacing itself is `min_unanchored_width`, the narrowest piece
    the scan found without an anchor, which should stay well above
    scan_spacing. The cost of a sweep is about (n + 1) * (cap width /
    scan_spacing) slack LPs.

    `capped` counts pieces of I(U) that end at a bound of log_nu_bounds, once
    per sweep.

    Parameters
    ----------
    grid_size : number of log-nu points laid over each piece to find the U_j range.
    hr_steps : hit-and-run steps per beta draw.
    beta_box : half-width of the box |beta_k| <= beta_box imposed in every LP.
        The sampler targets Q(y, U) intersected with this box. It changes
        nothing when Q is bounded and lies inside the box (the application's
        all-NB bins), but for sparse or zero-heavy data where Q is unbounded it
        restricts the parameter space, and the result depends on beta_box.
        Choose it so that |X beta| <= beta_box * ||X||_1 is far outside any
        plausible linear predictor, and report it alongside results for bins
        where the polytope is unbounded (a direction with no y > 0 support).
    log_nu_bounds : the analogous cap on log nu; see the module docstring.
    edge_tol, value_tol : tolerances of the golden-section refinement of the
        U_j range, in grid spacings and as a fraction of the range.
    exact_ranges : refine the U_j range between grid points (see
        _refine_extreme); False uses the grid values only.
    scan_spacing : spacing in log nu of the scan that locates feasible pieces.
    max_center_tries : redraws of log nu allowed when the center LP fails.
    """

    def __init__(self, X, y, t, beta_init, nu_init, seed=0, grid_size=15,
                 hr_steps=60, beta_box=100.0, log_nu_bounds=(-12.0, 12.0),
                 beta_cov=None, adapt_after=None, edge_tol=1.0 / 256,
                 value_tol=1e-6, exact_ranges=True, scan_spacing=0.1,
                 max_center_tries=20):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        self.t = np.asarray(t, float)
        self.rng = np.random.default_rng(seed)
        if int(grid_size) < 2:
            raise ValueError("grid_size must be at least 2")
        self.grid_size = int(grid_size)
        self.max_center_tries = int(max_center_tries)
        self.hr_steps = int(hr_steps)
        self.box = float(beta_box)
        self.log_nu_lo, self.log_nu_hi = map(float, log_nu_bounds)
        self.edge_tol = float(edge_tol)         # extreme tolerance, in grid spacings
        self.value_tol = float(value_tol)       # extreme tolerance, fraction of the U_j range
        # exact_ranges=False evaluates U_j ranges on the grid only. Adequate when
        # the feasible sets are narrow relative to the grid spacing; on sparse
        # data it can understate ranges by ~1%, so use exact_ranges=True for
        # small-count validation runs.
        self.exact_ranges = bool(exact_ranges)
        n_scan = max(3, int(np.ceil((self.log_nu_hi - self.log_nu_lo) / scan_spacing)) + 1)
        self._scan = np.linspace(self.log_nu_lo, self.log_nu_hi, n_scan)
        self.scan_spacing = float(self._scan[1] - self._scan[0])
        K = self.X.shape[1]

        # The starting nu must lie inside the cap on its own. Clipping it there
        # instead draws the initial uniforms from cells at a nu the data do not
        # support: with a fitted mean far from the counts the cell probabilities
        # underflow and Q(y, U) is empty from the first sweep, which surfaces
        # only as every sweep failing. A bin whose MLE falls outside the cap has
        # a cap-dependent fiducial distribution in any case, so stop here.
        if not self.log_nu_lo < float(np.log(nu_init)) < self.log_nu_hi:
            raise ValueError(
                f"starting log nu {float(np.log(nu_init)):.3f} is outside the cap "
                f"({self.log_nu_lo}, {self.log_nu_hi}); widen log_nu_bounds or check the "
                f"preliminary fit (beta_init {np.asarray(beta_init).round(3)})")
        r0 = np.exp(self.X @ beta_init) * self.t
        p0 = nu_init / (1 + nu_init)
        lo = nb_cdf(self.y - 1, r0, p0)
        hi = nb_cdf(self.y, r0, p0)
        self.U = np.clip(self.rng.uniform(lo, np.maximum(hi, lo + 1e-12)),
                         1e-15, 1 - 1e-15)
        self.log_nu = float(np.log(nu_init))   # a point known to be feasible

        # counters
        self.U_updates = 0
        self.skipped_updates = 0                # U_j left unchanged: no feasible piece found
        self.reverted_sweeps = 0                # sweeps undone: no feasible nu for the new U
        self.capped = 0                         # feasible sets ending at log_nu_bounds
        self.refinements = 0                    # points added locating interior extremes
        self.box_hits = 0                       # draws that reached the |beta| <= box truncation
        self.loo_multi = 0                      # U_j updates whose leave-one-out set was split
        self.multi_component = 0                # sweeps ending with I(U) split
        self.anchor_misses = 0                  # no feasible log nu located for the new U
        self.center_retries = 0                 # log nu redrawn: center LP failed there
        self.beta_failures = 0                  # sweeps with no draw: every redraw failed
        self.edge_disagreements = 0             # piece ends where the scan and the re-solve disagreed
        self.min_piece_width = np.inf           # narrowest piece of I(U) found
        self.narrow_pieces = 0                  # pieces of I(U) narrower than scan_spacing (found via an anchor)
        self.min_piece_width_loo = np.inf       # the same for the leave-one-out sets
        self.narrow_pieces_loo = 0
        self.min_unanchored_width = np.inf      # narrowest piece found by the scan alone

        # The eta bounds give every observation an upper bound but only y_i > 0
        # a lower one, so a direction along which the beta-set is unbounded must
        # lie in the null space of the positive-count rows. If those rows span,
        # the set is bounded by the data at every nu and beta_box never binds;
        # if they do not, the fiducial density depends on that constant and the
        # bin should not be reported without saying so.
        self.unbounded_by_design = bool(
            np.linalg.matrix_rank(self.X[self.y > 0]) < self.X.shape[1]
            if np.any(self.y > 0) else True)
        self._evals = []
        self._record_refined = None
        self._logR = None                       # warm start for r_solve
        self._beta_prev = None                  # start for hit-and-run

        # direction shape for hit-and-run: NB Fisher information at the start,
        # (X' diag(r_i p) X)^{-1}, replaced by a blend with the sample covariance
        # of the chain's own beta draws once enough have accumulated.
        if beta_cov is None:
            w = r0 * p0
            beta_cov = np.linalg.inv(self.X.T @ (self.X * w[:, None]))
        self._cov0 = np.asarray(beta_cov, float)
        self._L = np.linalg.cholesky(self._cov0)
        self.adapt_after = 10 * K if adapt_after is None else int(adapt_after)
        self._sum = np.zeros(K)
        self._sumsq = np.zeros((K, K))
        self._ndraws = 0

        # persistent LP models
        self.slack_lp = CenterLP(self.X, self.box, "slack")
        self.center_lp = CenterLP(self.X, self.box, "center")
        self.ext_lp = RangeLP(self.X, self.box)

        # The initial uniforms are drawn from the cells at (beta_init, nu_init),
        # so Q(y, U) contains that point unless the cell probabilities underflow.
        # Check rather than discover it as every sweep failing.
        slack = self._slack(self.log_nu)
        if not slack < 0:
            raise ValueError(
                f"initial uniforms are consistent with no beta at log nu "
                f"{self.log_nu:.3f} (slack {slack:.3g}): the starting fit "
                f"(beta {np.asarray(beta_init).round(3)}, log nu {self.log_nu:.3f}) "
                f"puts the counts in cells too narrow to represent")

    # -- geometry at a given nu -------------------------------------------------

    def _bounds(self, log_nu):
        lo, hi, self._logR = eta_bounds(self.U, self.y, self.t, np.exp(log_nu),
                                        guess=self._logR, return_logR=True)
        return lo, hi

    def _center(self, log_nu):
        lo, hi = self._bounds(log_nu)
        return rescaled_center(self.center_lp, lo, hi), lo, hi

    # -- locating feasible log nu ------------------------------------------------
    #
    # The exact Gibbs conditional of U_j given the other uniforms is uniform on
    #     A_j = { u_j : P_U(lambda) is nonempty for SOME lambda in the cap },
    # the union over the whole leave-one-out feasible log-nu set of the per-lambda
    # intervals A_j(lambda). That set can have several pieces. Within one piece
    # A_j(lambda) is an interval moving continuously with lambda, so its union
    # over the piece is one interval (smallest lower end to largest upper end);
    # A_j is the union of those intervals over the pieces, which need not be an
    # interval. The pieces are found by scanning the whole cap range, so nothing
    # here depends on the previous log nu except through `anchor`, a log nu known
    # to be feasible that is added to the scan so that a piece narrower than the
    # scan spacing is not lost when it is the one the chain is in. Pieces
    # narrower than the spacing that contain no anchor are not resolved.

    def _slack(self, lnu, drop=None, bounds=None):
        """Smallest uniform slack s with lower - s <= X beta <= upper + s at log
        nu, optionally without observation `drop`. Negative iff the polytope
        has interior; continuous in nu, so it can be root-found and minimized
        without stepping over a narrow set."""
        lo, hi = self._bounds(lnu) if bounds is None else bounds
        return _slack_value(self.slack_lp, lo, hi, drop)

    def _pieces(self, drop=None, anchors=(), count_cap=False):
        """All pieces (a, c) of {lambda in cap : P_U(lambda) nonempty}, without
        observation `drop` if given (feasible_pieces), with the bookkeeping."""
        pieces, info = feasible_pieces(self.U, self.y, self.t, self.slack_lp,
                                       self._scan, anchors, drop)
        self.edge_disagreements += info["edge_disagreements"]
        for (a, c), capped, anchored in zip(pieces, info["capped"], info["anchored"]):
            if count_cap and capped:
                self.capped += 1
            # resolution check, recorded separately for I(U) and for the
            # leave-one-out sets
            if drop is None:
                self.min_piece_width = min(self.min_piece_width, c - a)
                self.narrow_pieces += int(c - a < self.scan_spacing)
            else:
                self.min_piece_width_loo = min(self.min_piece_width_loo, c - a)
                self.narrow_pieces_loo += int(c - a < self.scan_spacing)
            # The scan has to find only pieces holding no anchor (a piece holding
            # one is found however narrow). Their narrowest width, compared with
            # scan_spacing, is the evidence that the spacing is fine enough.
            if not anchored:
                self.min_unanchored_width = min(self.min_unanchored_width, c - a)
        return pieces

    def _find_feasible(self, candidates):
        """A log nu near the candidates at which Q(y, U) is nonempty, or None:
        the best candidate by slack, then golden-section search on the slack
        between its neighbours."""
        cand = np.sort(np.asarray(candidates, float))
        v = np.array([self._slack(c) for c in cand])
        i = int(np.argmin(v))
        if v[i] < 0:
            return float(cand[i])
        lo_c = cand[max(i - 1, 0)]
        hi_c = cand[min(i + 1, len(cand) - 1)]
        if hi_c == lo_c:
            lo_c, hi_c = cand[i] - 0.05, cand[i] + 0.05
        gr = 0.5 * (np.sqrt(5) - 1)
        x1, x2 = hi_c - gr * (hi_c - lo_c), lo_c + gr * (hi_c - lo_c)
        f1, f2 = self._slack(x1), self._slack(x2)
        for _ in range(60):
            if min(f1, f2) < 0:
                return float(x1 if f1 < f2 else x2)
            if f1 < f2:
                hi_c, x2, f2 = x2, x1, f1
                x1 = hi_c - gr * (hi_c - lo_c)
                f1 = self._slack(x1)
            else:
                lo_c, x1, f1 = x1, x2, f2
                x2 = lo_c + gr * (hi_c - lo_c)
                f2 = self._slack(x2)
        return None

    # -- the range of U_j ---------------------------------------------------------

    def _loo_eval(self, j, x, yj, tj, lnu):
        """Leave-one-out (smallest F(y_j - 1), largest F(y_j)) at log nu = lnu, or
        None if the leave-one-out polytope is empty there."""
        lo, hi = eta_bounds(self.U, self.y, self.t, np.exp(lnu))
        self.ext_lp.set_bounds(lo, hi)
        rr = self.ext_lp.range(x, drop=j)
        if rr is None:
            return None
        p = np.exp(lnu) / (1 + np.exp(lnu))
        return (float(nb_cdf(yj - 1, np.exp(rr[1]) * tj, p)),
                float(nb_cdf(yj, np.exp(rr[0]) * tj, p)))

    def _refine_extreme(self, j, x, yj, tj, which, tol, lows, highs):
        """Golden-section search for the extreme of F between evaluated points.

        F(y_j) and F(y_j - 1) need not be monotone in nu over a leave-one-out
        piece, so on a coarse grid the best evaluated point can miss an interior
        extreme. which = 0 minimizes F(y_j - 1), which = 1 maximizes F(y_j).
        Every local extreme among the evaluated points is refined, not only the
        best one: with two interior extremes the global one can lie between
        grid points next to the one that is not best on the grid. Each bracket
        is the local extreme's neighbours among the points evaluated for this
        observation on the current piece (_u_interval_on resets _evals per
        piece); infeasible trial points are rejected.
        """
        pts = sorted(self._evals, key=lambda e: e[0])
        sgn = 1.0 if which == 0 else -1.0
        vals = {i: sgn * e[1][which] for i, e in enumerate(pts) if e[1] is not None}
        if len(vals) < 2:
            return
        nbrs = lambda i: [k for k in (i - 1, i + 1) if k in vals]
        local = [i for i in vals if all(vals[i] <= vals[k] for k in nbrs(i))]
        span = max(max(highs) - min(lows), 1e-300)
        vtol = self.value_tol * span
        for i_ext in local:
            # the extreme between two evaluated points cannot beat this point by
            # much more than its neighbours differ from it; skip when that is
            # below the value tolerance (the usual case when the set is narrow)
            nb_vals = [vals[k] for k in nbrs(i_ext)]
            if nb_vals and max(abs(v - vals[i_ext]) for v in nb_vals) <= vtol:
                continue
            self._golden(j, x, yj, tj, which, sgn, tol, vtol, lows, highs,
                         pts[max(i_ext - 1, 0)][0], pts[min(i_ext + 1, len(pts) - 1)][0])

    def _golden(self, j, x, yj, tj, which, sgn, tol, vtol, lows, highs, a, b):
        """Golden-section search for the extreme of sgn * F on (a, b)."""
        def obj(v):
            out = self._loo_eval(j, x, yj, tj, v)
            if out is None:
                return np.inf
            lows.append(out[0])
            highs.append(out[1])
            self.refinements += 1
            if self._record_refined is not None:
                self._record_refined.append((v, out))
            return sgn * out[which]
        gr = 0.5 * (np.sqrt(5.0) - 1.0)
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = obj(c), obj(d)
        while abs(b - a) > tol and not (np.isfinite(fc) and np.isfinite(fd)
                                         and abs(fc - fd) <= vtol):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = obj(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = obj(d)

    def _u_interval_on(self, j, a, c):
        """Union over lambda in the piece (a, c) of the values of U_j allowed
        by the other observations: (smallest F(y_j - 1), largest F(y_j)). Also
        returns the evaluated (lambda, (lo, hi)) pairs."""
        x, yj, tj = self.X[j], self.y[j], self.t[j]
        w = c - a
        # points just inside the ends: the extremes of F often sit at an edge
        grid = np.linspace(a + 1e-7 * w, c - 1e-7 * w, self.grid_size)
        nus = np.exp(grid)
        p_all = nus / (1 + nus)
        lower, upper = eta_bounds(self.U, self.y, self.t, nus)
        lows, highs = [], []
        self._evals = []
        for g in range(len(grid)):
            self.ext_lp.set_bounds(lower[g], upper[g])
            rr = self.ext_lp.range(x, drop=j)
            if rr is None:
                self._evals.append((grid[g], None))
                continue
            lo_v = float(nb_cdf(yj - 1, np.exp(rr[1]) * tj, p_all[g]))
            hi_v = float(nb_cdf(yj, np.exp(rr[0]) * tj, p_all[g]))
            lows.append(lo_v)
            highs.append(hi_v)
            self._evals.append((grid[g], (lo_v, hi_v)))
        if not lows:
            return None
        self._record_refined = []
        if self.exact_ranges:
            # interior extremes of F between grid points
            tol = (grid[1] - grid[0]) * self.edge_tol
            self._refine_extreme(j, x, yj, tj, 0, tol, lows, highs)
            self._refine_extreme(j, x, yj, tj, 1, tol, lows, highs)
        evals = [e for e in self._evals if e[1] is not None] + self._record_refined
        self._record_refined = None
        return min(lows), max(highs), evals

    def _uniform_on_union(self, intervals):
        """Uniform draw on a union of intervals, or None if it has length 0."""
        iv = sorted(intervals)
        merged = [list(iv[0])]
        for lo, hi in iv[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        lens = np.array([max(hi - lo, 0.0) for lo, hi in merged])
        if lens.sum() <= 0:
            return None
        k = int(self.rng.choice(len(merged), p=lens / lens.sum()))
        return float(self.rng.uniform(*merged[k]))

    def _direction_shape(self):
        """Cholesky factor of the current direction covariance."""
        if self._ndraws >= self.adapt_after:
            m = self._sum / self._ndraws
            cov = self._sumsq / self._ndraws - np.outer(m, m)
            # blend with the Fisher shape (normalized to equal trace) for stability
            cov = 0.5 * cov / np.trace(cov) + 0.5 * self._cov0 / np.trace(self._cov0)
            try:
                return np.linalg.cholesky(cov + 1e-12 * np.trace(cov) * np.eye(len(m)))
            except np.linalg.LinAlgError:
                pass
        return self._L

    # -- one Gibbs sweep -----------------------------------------------------------

    def _revert(self, U_entry, lnu_entry):
        """Undo a sweep that ended with no locatable feasible nu. With exact
        conditionals every update keeps U in the good set, so this is a guard
        against numerical failure only; `reverted_sweeps` should stay at 0."""
        self.U = U_entry
        self.log_nu = lnu_entry
        self.reverted_sweeps += 1

    def sweep(self):
        """One Gibbs sweep: the exact U_j conditionals, then log nu uniform on
        the whole of I(U) and beta uniform on its polytope. Returns (beta, nu,
        |I(U)|), or None if the sweep was undone."""
        U_entry, lnu_entry = self.U.copy(), self.log_nu
        anchor = self.log_nu        # drawn from I(U) last sweep, so feasible now
        for j in range(len(self.y)):
            pieces = self._pieces(drop=j, anchors=[anchor])
            if len(pieces) > 1:
                self.loo_multi += 1
            intervals, evals, owners = [], [], []
            for a, c in pieces:
                out = self._u_interval_on(j, a, c)
                if out is not None:
                    intervals.append(out[:2])
                    evals.extend(out[2])
                    owners.append((a, c, out[0], out[1]))
            uj = self._uniform_on_union(intervals) if intervals else None
            if uj is None:
                # No allowed U_j located. This cannot happen in exact arithmetic
                # (the anchor is feasible for the leave-one-out set), and
                # leaving U_j unchanged is a numerical guard; `skipped_updates`
                # should stay at 0.
                self.skipped_updates += 1
                continue
            self.U[j] = np.clip(uj, 1e-15, 1 - 1e-15)
            self.U_updates += 1
            # a log nu at which the new U is feasible, for the next scan
            hit = [lam for lam, (lo, hi) in evals if lo < self.U[j] <= hi]
            if hit:
                anchor = hit[len(hit) // 2]
            else:
                # U_j lies in the union of A_j(lambda) over a piece whose
                # interval contains it, so the full set is nonempty somewhere in
                # that piece, between evaluated points: search there
                cand = [lam for a, c, lo, hi in owners if lo < self.U[j] <= hi
                        for lam in np.linspace(a, c, self.grid_size)]
                found = self._find_feasible(cand) if cand else None
                if found is not None:
                    anchor = found
                else:
                    self.anchor_misses += 1
        pieces = self._pieces(anchors=[anchor], count_cap=True)
        if not pieces:
            self._revert(U_entry, lnu_entry)
            return None
        if len(pieces) > 1:
            self.multi_component += 1
        lens = np.array([c - a for a, c in pieces])
        # Selection: log nu uniform on I(U), then beta uniform on P_U(log nu).
        # The Chebyshev-center LP that starts hit-and-run can fail numerically
        # where P_U(log nu) is extremely thin (next to an end of a piece); log nu
        # is then redrawn, which is rejection sampling from the uniform on I(U)
        # restricted to where the LP succeeds.
        for _ in range(self.max_center_tries):
            k = int(self.rng.choice(len(pieces), p=lens / lens.sum()))
            lnu = self.rng.uniform(*pieces[k])
            cc, lo, hi = self._center(lnu)
            if cc is not None and cc[1] > 0:
                return self._select_beta(lnu, cc, lo, hi, float(lens.sum()))
            self.center_retries += 1
        # U is still a valid state (I(U) is nonempty), so keep it; only this
        # sweep's draw is lost. The next sweep needs a log nu feasible for U:
        # the middle of the longest piece is one.
        a, c = pieces[int(np.argmax(lens))]
        self.log_nu = 0.5 * (a + c)
        self.beta_failures += 1
        return None

    def _select_beta(self, lnu, cc, lo, hi, width):
        """Second half of the selection rule at the drawn log nu: beta uniform
        on the polytope P_U(lnu), by hit-and-run from the previous draw when it
        lies inside the polytope and otherwise from the Chebyshev center cc."""
        self.log_nu = lnu
        A, bb = _halfspaces(self.X, lo, hi, self.box)
        # start from the previous draw when it lies inside the new polytope
        x0 = cc[0]
        if self._beta_prev is not None and np.all(A @ self._beta_prev < bb):
            x0 = self._beta_prev
        beta = hit_and_run(A, bb, x0, self.hr_steps, self.rng, L=self._direction_shape())
        if np.max(np.abs(beta)) > self.box * (1 - 1e-8):
            self.box_hits += 1      # drawn at the truncation, not inside the data set
        self._beta_prev = beta
        self._sum += beta
        self._sumsq += np.outer(beta, beta)
        self._ndraws += 1
        return beta, float(np.exp(lnu)), width

    def run(self, sweeps, progress=None):
        """Draws from `sweeps` sweeps; rows of undone sweeps are NaN."""
        K = self.X.shape[1]
        betas = np.full((sweeps, K), np.nan)
        nus = np.full(sweeps, np.nan)
        widths = np.full(sweeps, np.nan)
        for s in range(sweeps):
            out = self.sweep()
            if out is not None:
                betas[s], nus[s], widths[s] = out
            if progress is not None:
                progress(s)
        return betas, nus, widths
