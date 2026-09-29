"""MH sampler for the discrete generalized fiducial distribution of one
entirely-NB component (Supplement, Section 8.3, Algorithm 1).

Targets the same distribution as the Gibbs sampler discrete_gfi.DiscreteGFI
(selection V: log nu uniform on I(U), then beta uniform on P_U(log nu)), but
moves theta and the uniforms together. State (theta, W), with W_i in (0, 1) the
relative position of U_i in its cell,
    U_i = F(y_i - 1; theta) + W_i f(y_i; theta),
so the density of (theta, W) is proportional to L(theta) v_U(theta), with
    v_U(theta) = 1 / (|I(U)| vol P_U(log nu)).
  theta-move: theta' from an independence t proposal (df 5, scale 1.3^2 times
      the inverse observed information at the MLE), W carried along; accept
      with the ratio of L(theta) v_U(theta) times the proposal ratio.
  W-move: redraw a block of max(1, n/10) of the W_i; accept with the ratio of v.

|I(U)| comes from discrete_gfi.feasible_pieces, with the state's own log nu as
anchor, so v_U(theta) is a deterministic function of the state and the chain is
exact for that target; the only approximation is a piece of I(U) away from the
anchor and narrower than the scan spacing (`min_unanchored_width`).
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull, QhullError
from scipy.special import gammaln
from scipy.stats import multivariate_t

from discrete_gfi import (nb_cdf, eta_bounds, CenterLP, feasible_pieces, _slack_value,
                          rescaled_center)


def nb_loglik(X, y, t, beta, log_nu):
    nu = np.exp(log_nu)
    eta = X @ beta
    r = np.exp(eta) * t
    p = nu / (1 + nu)
    return float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                        + r * np.log(p) + y * np.log1p(-p)))


def observed_info(X, y, t, beta, log_nu, h=1e-4):
    """Numerical Hessian of -loglik in (beta, log nu)."""
    th = np.concatenate([beta, [log_nu]])
    d = len(th)
    f = lambda v: -nb_loglik(X, y, t, v[:-1], v[-1])
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            e_i, e_j = np.eye(d)[i] * h, np.eye(d)[j] * h
            H[i, j] = (f(th + e_i + e_j) - f(th + e_i - e_j)
                       - f(th - e_i + e_j) + f(th - e_i - e_j)) / (4 * h * h)
            H[j, i] = H[i, j]
    return H


class ConsistentSet:
    """Geometry of Q(y, U): the feasible log-nu set and the beta-polytope volume.

    Counters (all should be 0 or near it, except volume_calls and the widths):
      anchor_infeasible  the evaluated log nu was not inside any piece found,
                         although it is feasible in exact arithmetic;
                         anchor_missed_with_interior counts those where the
                         slack at log nu was negative, which would be a bug
      degenerate_width   the pieces found had total length 0
      vol_fail_center / vol_fail_qhull
                         the volume could not be computed; vol_fail_feasible
                         counts those where the slack said the set had interior,
                         i.e. false rejections, which fall on thin polytopes
                         (the largest v_U) and so bias the chain if frequent
      vol_joggled        volumes qhull computed only with option QJ, which
                         perturbs its input; should be 0 since the center is
                         re-solved at the polytope's own scale
      edge_disagreements piece ends where the scan and the re-solve disagreed
      box_hits           volumes truncated by the box |beta_k| <= box
    and, for resolution, min_unanchored_width: the narrowest piece the scan
    found without an anchor, which should stay well above scan_spacing.
    """

    def __init__(self, X, y, t, beta_hat, cov_beta, log_nu_bounds=(-12., 12.),
                 box=100.0, scan_spacing=0.02):
        self.X, self.y, self.t = X, y, t
        self.n, self.k = X.shape
        self.lo_cap, self.hi_cap = log_nu_bounds
        n_scan = max(3, int(np.ceil((self.hi_cap - self.lo_cap) / scan_spacing)) + 1)
        self._scan = np.linspace(self.lo_cap, self.hi_cap, n_scan)
        self.scan_spacing = float(self._scan[1] - self._scan[0])
        self.beta_hat = beta_hat
        self.box = box
        # whitening: beta = beta_hat + L z. Volumes in z differ from volumes in
        # beta by the constant det L, which cancels in every ratio.
        self.L = np.linalg.cholesky(cov_beta)
        self.A = X @ self.L
        self.slack_lp = CenterLP(X, box, "slack")
        # The Chebyshev center is taken over the data rows AND the |beta| <= box
        # rows: CenterLP's own box is a plain bound on the variables, so without
        # the extra rows the center can land on a box face and the inscribed
        # radius overstate the set, leaving no interior point for qhull.
        self.center_lp = CenterLP(np.vstack([self.A, self.L]), 1e8, "center")
        self.volume_calls = 0
        self.box_hits = 0
        self.anchor_infeasible = 0
        self.anchor_missed_with_interior = 0    # of those, slack < 0 at lnu: a bug if ever > 0
        self.degenerate_width = 0
        self.vol_fail_center = 0
        self.vol_fail_qhull = 0
        self.vol_fail_feasible = 0
        self.vol_joggled = 0        # volumes qhull computed only with QJ (a perturbed input)
        self.edge_disagreements = 0
        self.min_unanchored_width = np.inf
        # Structural check: the eta bounds give an upper bound for every
        # observation but a lower bound only where y_i > 0, so an unbounded
        # direction must lie in the null space of the positive-count rows. If
        # those rows span, the box never binds; if they do not, the fiducial
        # density depends on the box constant at every nu.
        yv = np.asarray(y)
        self.unbounded_by_design = bool(
            np.linalg.matrix_rank(np.asarray(X)[yv > 0]) < self.k
            if np.any(yv > 0) else True)

    def _violation(self, U, lnu):
        """Uniform-slack value at log nu (negative iff the beta-set has interior)."""
        return _slack_value(self.slack_lp, *eta_bounds(U, self.y, self.t, np.exp(lnu)))

    # -- the feasible log-nu set --------------------------------------------------
    def pieces(self, U, lnu):
        """Pieces of I(U) (feasible_pieces, with lnu as the anchor) and the
        bookkeeping."""
        pieces, info = feasible_pieces(U, self.y, self.t, self.slack_lp,
                                       self._scan, anchors=[lnu])
        self.edge_disagreements += info["edge_disagreements"]
        for (a, c), anchored in zip(pieces, info["anchored"]):
            if not anchored:
                self.min_unanchored_width = min(self.min_unanchored_width, c - a)
        return pieces, info

    # -- polytope volume ------------------------------------------------------
    def log_volume(self, U, lnu, bounds=None):
        """log vol of {z : lo < A z + X beta_hat <= hi, |beta| <= box} (whitened
        beta). The box matters only when fewer than k observations have a
        positive count, in which case the eta bounds leave the beta-set
        unbounded; it is the same truncation DiscreteGFI applies, so the two
        samplers normalize over the same set. `bounds`, if given, is the pair
        (lo, hi) of eta bounds at lnu, used instead of computing them from U
        (check_failed_proposals.py passes high-precision bounds)."""
        self.volume_calls += 1
        lo, hi = eta_bounds(U, self.y, self.t, np.exp(lnu)) if bounds is None else bounds
        off = self.X @ self.beta_hat
        b_hi = hi - off
        fin = np.isfinite(lo)
        b_lo = lo[fin] - off[fin]
        # |beta| <= box in whitened coordinates: +-L z <= box -+ beta_hat
        Abox = np.vstack([self.L, -self.L])
        bbox = np.concatenate([self.box - self.beta_hat, self.box + self.beta_hat])
        if self.k == 1:
            ad, cd = self.A[:, 0], b_hi
            up = np.min(np.where(ad > 0, cd / ad, np.inf))
            lw = np.max(np.where(ad > 0, -np.inf, cd / ad))
            if fin.any():
                af = self.A[fin, 0]
                up = min(up, np.min(np.where(af < 0, b_lo / af, np.inf)))
                lw = max(lw, np.max(np.where(af > 0, b_lo / af, -np.inf)))
            ab, cb = Abox[:, 0], bbox
            up_b = np.min(np.where(ab > 0, cb / ab, np.inf))
            lw_b = np.max(np.where(ab > 0, -np.inf, cb / ab))
            if up_b < up or lw_b > lw:
                self.box_hits += 1                  # the box, not the data, sets an end
            up, lw = min(up, up_b), max(lw, lw_b)
            return np.log(up - lw) if up > lw else -np.inf
        # Chebyshev center, re-solved at the polytope's own scale (discrete_gfi.
        # rescaled_center): at large counts the inscribed radius can fall below
        # 1e-6, and a center accurate only to the LP's absolute tolerance may
        # lie outside a face, which qhull rejects.
        cc = rescaled_center(self.center_lp,
                             np.concatenate([lo - off, -self.box - self.beta_hat]),
                             np.concatenate([hi - off, self.box - self.beta_hat]))
        if cc is None:
            self.vol_fail_center += 1
            if _slack_value(self.slack_lp, lo, hi) < 0:
                self.vol_fail_feasible += 1
            return -np.inf
        z0, rad = cc
        # rescale so the inscribed radius is 1: keeps qhull well conditioned
        Hs = np.vstack([np.hstack([self.A, -(b_hi - self.A @ z0)[:, None] / rad]),
                        np.hstack([-self.A[fin], (b_lo - self.A[fin] @ z0)[:, None] / rad]),
                        np.hstack([Abox, -(bbox - Abox @ z0)[:, None] / rad])])
        for opts in ("Qx", "QJ"):
            try:
                hs = HalfspaceIntersection(Hs, np.zeros(self.k), qhull_options=opts)
                vol = ConvexHull(hs.intersections).volume
                # a vertex on the box means the set is truncated there rather
                # than bounded by the data: the volume, and so v_U, depends on
                # the box constant
                vert_beta = self.beta_hat + (z0 + rad * hs.intersections) @ self.L.T
                if np.max(np.abs(vert_beta)) > self.box * (1 - 1e-8):
                    self.box_hits += 1
                if opts == "QJ":
                    self.vol_joggled += 1
                return np.log(vol) + self.k * np.log(rad)
            except QhullError:
                continue
        self.vol_fail_qhull += 1
        if _slack_value(self.slack_lp, lo, hi) < 0:
            self.vol_fail_feasible += 1
        return -np.inf

    def log_v(self, U, beta, lnu):
        """log v_U(theta) = -log |I(U)| - log vol P_U(nu), and a summary of
        I(U): its total length, number of pieces, and whether a piece reaches
        the cap. Returns (-inf, None) when the geometry cannot be evaluated,
        which rejects the proposal."""
        pieces, info = self.pieces(U, lnu)
        # theta is in Q(y, U) by construction, so lnu lies in a piece; if the
        # search did not put it in one, the geometry failed numerically here.
        # Two cases, told apart by the slack at lnu itself: no interior to the
        # LP's tolerance (a thin state, e.g. a u_i rounded onto a cell end), or
        # interior present yet no piece holding lnu, which would be a bug in
        # feasible_pieces.
        if not any(a - 1e-12 <= lnu <= c + 1e-12 for a, c in pieces):
            self.anchor_infeasible += 1
            if self._violation(U, lnu) < 0:
                self.anchor_missed_with_interior += 1
            return -np.inf, None
        total = sum(c - a for a, c in pieces)
        if total <= 0:
            self.degenerate_width += 1
            return -np.inf, None
        summary = dict(width=total, n_pieces=len(pieces), capped=any(info["capped"]),
                       min_unanchored=min((c - a for (a, c), anc in
                                           zip(pieces, info["anchored"]) if not anc),
                                          default=np.inf))
        return -np.log(total) - self.log_volume(U, lnu), summary


class ThetaMH:
    """State (theta, W) with W_i in (0,1) the relative position of U_i within
    its cell: U_i = F(y_i - 1; theta) + W_i f(y_i; theta).

    Moves (module docstring): theta' from the independence t proposal with W
    carried along, then a block of W redrawn at fixed theta.

    Per-state counters, updated once per step on the state the chain holds
    after both moves (comparable with the Gibbs sampler's per-sweep counts and
    with the reference's fraction of feasible u): `states`, `split_states`
    (I(U) has more than one piece) and `capped_states` (a piece reaches the
    cap). The geometry's own counters are on `self.geom`.

    Proposals whose target could not be evaluated (-inf) are rejected and
    counted: `inf_theta` and `inf_W` by move. For theta proposals the log
    acceptance ratio without its v factor is recorded (`inf_theta_max_gap`,
    the largest), and with it the log v each would have needed to be accepted
    (`inf_theta_min_required_logv`, the smallest). v grows at a tail proposal,
    so the gap alone does not settle whether a rejection mattered; the required
    log v compared with `max_state_logv`, the largest log v of any state the
    chain held, does. Any `inf_W` is a false rejection, since theta is unchanged.
    """

    def __init__(self, X, y, t, beta_hat, nu_hat, seed=0, t_df=5, t_scale=1.3,
                 w_block=None, log_nu_bounds=(-12., 12.), scan_spacing=0.02,
                 record_failures=0):
        # record_failures > 0 keeps up to that many theta proposals rejected
        # because their target could not be evaluated, with what is needed to
        # recompute them exactly (check_failed_proposals.py); 0 keeps none.
        self.record_failures = int(record_failures)
        self.failures = []
        self.X, self.y, self.t = (np.asarray(v, float) for v in (X, y, t))
        self.rng = np.random.default_rng(seed)
        self.n, self.k = self.X.shape
        self.theta_hat = np.concatenate([beta_hat, [np.log(nu_hat)]])
        H = observed_info(self.X, self.y, self.t, beta_hat, np.log(nu_hat))
        self.Sigma = np.linalg.inv(H)
        self.geom = ConsistentSet(self.X, self.y, self.t, beta_hat,
                                  self.Sigma[:self.k, :self.k], log_nu_bounds,
                                  scan_spacing=scan_spacing)
        self.prop = multivariate_t(self.theta_hat, t_scale ** 2 * self.Sigma, df=t_df)
        self.w_block = max(1, self.n // 10) if w_block is None else int(w_block)
        self.acc = {"theta": 0, "W": 0}
        self.tries = {"theta": 0, "W": 0}
        self.states = 0
        self.split_states = 0
        self.capped_states = 0
        # proposals rejected because their target was -inf (geometry failed)
        self.inf_theta = 0
        self.inf_theta_max_gap = -np.inf              # largest log acceptance ratio without v
        self.inf_theta_min_required_logv = np.inf     # smallest log v that would have rescued one
        self.inf_theta_required = []                  # that log v for every such proposal
        self.inf_W = 0
        self.max_state_logv = -np.inf                 # largest log v of a state the chain held
        # narrowest piece found by the scan alone, over the states the chain
        # held (geom.min_unanchored_width also covers rejected proposals)
        self.min_state_unanchored_width = np.inf

    def _uniform01(self, size):
        """Uniform draws on the open interval (0, 1). Generator.random is on
        [0, 1); an exact 0 would put U_i on the excluded lower end of its cell,
        so the (probability 2^-53) zeros are redrawn."""
        w = self.rng.random(size)
        while np.any(w == 0.0):
            z = w == 0.0
            w[z] = self.rng.random(int(z.sum()))
        return w

    def cells(self, beta, lnu):
        r = np.exp(self.X @ beta) * self.t
        p = np.exp(lnu) / (1 + np.exp(lnu))
        lo, hi = nb_cdf(self.y - 1, r, p), nb_cdf(self.y, r, p)
        return lo, hi

    def U_of(self, theta, W):
        lo, hi = self.cells(theta[:-1], theta[-1])
        return lo + (hi - lo) * W

    def _log_target(self, theta, W):
        """(log target, summary, log-likelihood). The summary also carries the
        log-likelihood and log v separately, for the rejection diagnostics."""
        U = self.U_of(theta, W)
        beta, lnu = theta[:-1], theta[-1]
        lv, summary = self.geom.log_v(U, beta, lnu)
        ll = nb_loglik(self.X, self.y, self.t, beta, lnu)
        if summary is not None:
            summary = dict(summary, loglik=ll, logv=lv)
        return ll + lv, summary, ll

    def init(self, theta=None):
        theta = self.theta_hat.copy() if theta is None else np.asarray(theta, float)
        # Outside the cap the target is zero, so the chain would run from a
        # state it can never return to and reject nearly every proposal. The
        # starting log nu has to be strictly inside it; a bin whose MLE is not
        # has a cap-dependent fiducial distribution and should be reported as
        # capped rather than sampled.
        if not (self.geom.lo_cap < theta[-1] < self.geom.hi_cap
                and np.all(np.abs(theta[:-1]) < self.geom.box)):
            raise ValueError(
                f"starting point (beta {theta[:-1].round(3)}, log nu {theta[-1]:.3f}) is "
                f"outside Theta: |beta_k| < {self.geom.box} and log nu in "
                f"({self.geom.lo_cap}, {self.geom.hi_cap}); widen the bounds or check "
                f"the preliminary fit")
        for _ in range(20):
            W = self._uniform01(self.n)
            lt, summary, _ = self._log_target(theta, W)
            if np.isfinite(lt):
                break
        if not np.isfinite(lt):
            raise ValueError(
                f"no feasible start after 20 draws of W at theta "
                f"(beta {theta[:-1].round(3)}, log nu {theta[-1]:.3f}): the fit puts "
                f"the counts in cells too narrow to represent, or Q(y, U) is empty there")
        self.state = (theta, W, lt, summary)

    def step_theta(self):
        theta, W, lt, summary = self.state
        th_new = self.prop.rvs(random_state=self.rng)
        log_k = self.prop.logpdf(theta) - self.prop.logpdf(th_new)
        self.tries["theta"] += 1
        # The target is zero outside Theta. log_v truncates the polytope at the
        # box but never checks beta itself, so a proposal outside it must be
        # rejected here.
        if not (self.geom.lo_cap < th_new[-1] < self.geom.hi_cap
                and np.all(np.abs(th_new[:-1]) < self.geom.box)):
            return
        lt_new, summary_new, ll_new = self._log_target(th_new, W)
        if not np.isfinite(lt_new):
            # A proposal inside Theta whose geometry could not be evaluated is
            # rejected. It matters only if it could have been accepted, and v
            # grows at a tail proposal (its cells, |I(U)| and the volume all
            # shrink), which is the direction that could rescue it. So record
            # what it would have needed: accepting requires
            #     log v(theta') > log v(theta) - gap,
            # with gap the log acceptance ratio without its v factor (the
            # log-likelihood ratio plus the log proposal ratio). The smallest
            # such requirement is reported beside the largest log v the chain
            # ever held (`max_state_logv`); a requirement far above that means
            # the proposal would have been rejected in any case.
            self.inf_theta += 1
            gap = ll_new - summary["loglik"] + log_k
            required = summary["logv"] - gap
            self.inf_theta_max_gap = max(self.inf_theta_max_gap, gap)
            self.inf_theta_min_required_logv = min(self.inf_theta_min_required_logv, required)
            self.inf_theta_required.append(required)
            if len(self.failures) < self.record_failures:
                self.failures.append(dict(
                    theta_new=th_new.copy(), W=W.copy(), loglik_new=ll_new, log_k=log_k,
                    loglik_cur=summary["loglik"], logv_cur=summary["logv"],
                    theta_cur=theta.copy(), required_logv=required))
            return
        if np.log(self.rng.uniform()) < lt_new - lt + log_k:
            self.state = (th_new, W, lt_new, summary_new)
            self.acc["theta"] += 1

    def step_W(self):
        theta, W, lt, summary = self.state
        W_new = W.copy()
        idx = self.rng.choice(self.n, self.w_block, replace=False)
        W_new[idx] = self._uniform01(len(idx))
        self.tries["W"] += 1
        lt_new, summary_new, _ = self._log_target(theta, W_new)
        if not np.isfinite(lt_new):
            # same theta, so the likelihood is unchanged: a W proposal rejected
            # this way is a false rejection whenever it happens
            self.inf_W += 1
            return
        if np.log(self.rng.uniform()) < lt_new - lt:
            self.state = (theta, W_new, lt_new, summary_new)
            self.acc["W"] += 1

    def step(self):
        self.step_theta()
        self.step_W()
        summary = self.state[3]
        self.states += 1
        self.split_states += int(summary["n_pieces"] > 1)
        self.capped_states += int(summary["capped"])
        self.max_state_logv = max(self.max_state_logv, summary["logv"])
        self.min_state_unanchored_width = min(self.min_state_unanchored_width,
                                              summary["min_unanchored"])

    def run(self, n, progress=None):
        """n steps; returns the theta draws and |I(U)| (total length over all
        pieces) of the state after each step."""
        out = np.empty((n, self.k + 1))
        widths = np.empty(n)
        for i in range(n):
            self.step()
            out[i] = self.state[0]
            widths[i] = self.state[3]["width"]
            if progress is not None:
                progress(i)
        return out, widths
