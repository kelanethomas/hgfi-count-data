"""Exact reference for the discrete GFD at small n (Supplement, Section 8.4):
rejection sampling from the definition, for K = 1 or 2.

Given U, the feasible log-nu set is bracketed by a scan and each end located by
bisection to about 1e-13; log nu is drawn uniformly on the set (pieces in
proportion to their lengths), then beta uniformly on the polygon by
area-weighted triangulation. Feasibility is tested by vertex enumeration; the
code shares nothing with the samplers (discrete_gfi.py, gfd_theta_mh.py).
"""
import numpy as np
from scipy.special import betainc

TOL = 1e-9


def _r_solve(u, y, p, iters=45):
    """R with F(y; R, p) = I_p(R, y + 1) = u, by bisection on log R in (-40, 40)."""
    u, y, p = np.broadcast_arrays(np.asarray(u, float), np.asarray(y, float),
                                  np.asarray(p, float))
    lo, hi = np.full(u.shape, -40.0), np.full(u.shape, 40.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = betainc(np.exp(mid), y + 1.0, p) > u
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return np.exp(0.5 * (lo + hi))


class Reference:
    def __init__(self, X, y, t, box=100.0, lo_cap=-6.0, hi_cap=6.0):
        self.X, self.y, self.t = map(np.asarray, (X, y, t))
        self.n, self.k = self.X.shape
        self.box, self.lo_cap, self.hi_cap = box, lo_cap, hi_cap
        self.pos = self.y > 0
        # constraint normals, constant in nu:  A beta <= b(nu)
        self.A = np.vstack([self.X, -self.X[self.pos],
                            np.eye(self.k), -np.eye(self.k)])
        if self.k == 2:
            P = [(p, q) for p in range(len(self.A)) for q in range(p + 1, len(self.A))]
            M = np.array([[self.A[p], self.A[q]] for p, q in P])
            det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
            self.pairs = np.array(P)[np.abs(det) > 1e-12]
            M = M[np.abs(det) > 1e-12]
            self.Minv = np.linalg.inv(M)

    def bounds(self, U, lnu):
        """eta bounds (lo, hi) at one nu, or stacked over an array of nu."""
        nu = np.exp(np.atleast_1d(lnu))
        p = (nu / (1 + nu))[:, None]
        hi = np.log(_r_solve(U[None], self.y[None], p) / self.t)
        lo = np.where(self.y > 0,
                      np.log(_r_solve(U[None], np.maximum(self.y - 1, 0)[None], p) / self.t),
                      -np.inf)
        return lo, hi

    def rhs(self, lo, hi):
        """b(nu) rows matching self.A, shape (G, L)."""
        G = hi.shape[0]
        return np.concatenate([hi, -lo[:, self.pos],
                               np.full((G, 2 * self.k), self.box)], axis=1)

    def feasible(self, U, lnus):
        """Does the beta-set have positive volume at each of these nu? (G,) bool."""
        lo, hi = self.bounds(U, lnus)
        b = self.rhs(lo, hi)
        if self.k == 1:
            up = np.min(np.where(self.A[:, 0] > 0, b / self.A[:, 0], np.inf), axis=1)
            lw = np.max(np.where(self.A[:, 0] > 0, -np.inf, b / self.A[:, 0]), axis=1)
            return up - lw > TOL
        rhs2 = b[:, self.pairs]                              # (G, P, 2)
        V = np.einsum("pij,gpj->gpi", self.Minv, rhs2)       # candidate vertices
        ok = np.all(self.A @ V.transpose(0, 2, 1) <= b[:, :, None] + 1e-7, axis=1)  # (G, P)
        # positive area needs three distinct feasible vertices
        out = np.zeros(len(np.atleast_1d(lnus)), bool)
        for g in np.where(ok.any(1))[0]:
            pts = V[g][ok[g]]
            if len(pts) >= 3:
                c = pts - pts.mean(0)
                out[g] = np.linalg.matrix_rank(c, tol=1e-8) == 2 and self._area(pts) > TOL
        return out

    @staticmethod
    def _area(pts):
        c = pts.mean(0)
        o = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        x, v = pts[o, 0], pts[o, 1]
        return 0.5 * abs(x @ np.roll(v, -1) - v @ np.roll(x, -1))

    def intervals(self, U, scan_h=0.02):
        """Feasible log-nu intervals: a scan at spacing scan_h to bracket, then
        bisection for each end. Returns a list of (a, b) and the scan mask."""
        grid = np.arange(self.lo_cap, self.hi_cap + scan_h, scan_h)
        f = self.feasible(U, grid)
        if not f.any():
            return [], f
        out = []
        starts = np.where(f & ~np.r_[False, f[:-1]])[0]
        ends = np.where(f & ~np.r_[f[1:], False])[0]
        for s, e in zip(starts, ends):
            a = (self.lo_cap if s == 0 else
                 self._bisect(U, grid[s - 1], grid[s]))
            b = (self.hi_cap if e == len(grid) - 1 else
                 self._bisect(U, grid[e + 1], grid[e]))
            out.append((a, b))
        return out, f

    def _bisect(self, U, bad, good, iters=45):
        for _ in range(iters):
            mid = 0.5 * (bad + good)
            if self.feasible(U, np.array([mid]))[0]:
                good = mid
            else:
                bad = mid
        return good

    def draw(self, U, rng, scan_h=0.02):
        """One reference draw (beta, log nu) by the selection rule V, or None."""
        itvs, _ = self.intervals(U, scan_h)
        if not itvs:
            return None, itvs
        w = np.array([b - a for a, b in itvs])
        a, b = itvs[rng.choice(len(itvs), p=w / w.sum())]
        lnu = rng.uniform(a, b)
        lo, hi = self.bounds(U, np.array([lnu]))
        beta = self._uniform_beta(lo[0], hi[0], rng)
        return (None if beta is None else (beta, lnu)), itvs

    def _uniform_beta(self, lo, hi, rng):
        b = self.rhs(lo[None], hi[None])[0]
        if self.k == 1:
            up = np.min(np.where(self.A[:, 0] > 0, b / self.A[:, 0], np.inf))
            lw = np.max(np.where(self.A[:, 0] > 0, -np.inf, b / self.A[:, 0]))
            return None if up <= lw else np.array([rng.uniform(lw, up)])
        V = np.einsum("pij,pj->pi", self.Minv, b[self.pairs])
        ok = np.all(self.A @ V.T <= b[:, None] + 1e-7, axis=0)
        pts = V[ok]
        if len(pts) < 3:
            return None
        c = pts.mean(0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        tri = [(pts[0], pts[i], pts[i + 1]) for i in range(1, len(pts) - 1)]
        ar = np.array([self._area(np.array(T)) for T in tri])
        if ar.sum() <= 0:
            return None
        A_, B_, C_ = tri[rng.choice(len(tri), p=ar / ar.sum())]
        r1, r2 = rng.uniform(size=2); s = np.sqrt(r1)
        return (1 - s) * A_ + s * (1 - r2) * B_ + s * r2 * C_
