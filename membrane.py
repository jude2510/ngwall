"""Relativistic Nambu-Goto membrane (domain wall) evolver in flat 3+1 spacetime.

Gauge: X^0 = t, with transverse velocity (Xdot . d_i X = 0), which is
preserved by the dynamics if satisfied initially (e.g. starting from rest).
In this gauge the NG action is

    S = -sigma * int dt du dv sqrt((1 - Xdot^2) h),   h = det(d_i X . d_j X)

Discretization: linear elements on a triangle mesh with the discrete Lagrangian

    L = -sigma * sum_T A_T(X) * (1/3) sum_{a in T} sqrt(1 - V_a^2)

(A_T = current triangle area, V_a = vertex velocity, c = 1). Euler-Lagrange:

    p_a = sigma * Atil_a * gamma_a * V_a,   Atil_a = sum_{T ni a} A_T / 3
    dp_a/dt = -sigma * sum_{T ni a} s_T * grad_a A_T,
    s_T = (1/3) sum_{b in T} sqrt(1 - V_b^2)

i.e. a "relativistic surface tension" network: forces are area gradients
weighted by local time-dilation factors. The conserved energy is

    E = sum_a sqrt((sigma * Atil_a)^2 + |p_a|^2)  ->  sigma * int gamma dA,

the exact NG energy. Velocity recovery from momentum is closed-form:
V_a = p_a / sqrt((sigma*Atil_a)^2 + |p_a|^2). Time stepping: RK4 on (X, p).
"""

import numpy as np


def icosphere(nsub=4):
    """Unit icosphere: (vertices (N,3), faces (M,3)). nsub=4 -> 2562 verts."""
    phi = (1 + 5 ** 0.5) / 2
    V = np.array(
        [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
         [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
         [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(nsub):
        verts = list(V)
        cache = {}

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = verts[i] + verts[j]
                cache[key] = len(verts)
                verts.append(m / np.linalg.norm(m))
            return cache[key]

        newF = []
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V, F = np.array(verts), np.array(newF)
    return V, F


class Membrane:
    def __init__(self, X, faces, sigma=1.0):
        self.faces = np.asarray(faces)
        self.sigma = sigma
        self.X = np.array(X, float)
        self.p = np.zeros_like(self.X)
        self.t = 0.0
        self._build_adjacency()

    def _build_adjacency(self):
        """1-ring neighbor lists and 2-ring triangle candidates (flat arrays)."""
        n = len(self.X)
        nbr = [set() for _ in range(n)]
        vtri = [set() for _ in range(n)]
        for ti, (a, b, c) in enumerate(self.faces):
            nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
            vtri[a].add(ti); vtri[b].add(ti); vtri[c].add(ti)
        # neighbor lists as (flat idx, offsets) for vectorized centroids
        counts = np.array([len(s) for s in nbr])
        self.nbr_flat = np.concatenate([sorted(s) for s in nbr])
        self.nbr_off = np.concatenate([[0], np.cumsum(counts)])
        # candidate triangles for closest-point reprojection: 2-ring
        cand_v, cand_t = [], []
        for v in range(n):
            tris = set(vtri[v])
            for u in nbr[v]:
                tris |= vtri[u]
            cand_v += [v] * len(tris)
            cand_t += sorted(tris)
        self.cand_v = np.array(cand_v)
        self.cand_t = np.array(cand_t)

    # -- geometry ---------------------------------------------------------
    def tri_geom(self, X):
        """Areas (M,) and area gradients grad_a A_T (M, corner, 3)."""
        a, b, c = (X[self.faces[:, k]] for k in range(3))
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n, axis=1)
        # a zero-area triangle would make this 0/0 -> NaN, which spreads to
        # the entire mesh within ~3 RK4 steps; the clip gives it a zero normal
        # and hence zero area gradient -- no force, the correct limit.
        nhat = n / np.clip(norm, 1e-300, None)[:, None]
        gA = 0.5 * np.stack(
            [np.cross(nhat, c - b), np.cross(nhat, a - c), np.cross(nhat, b - a)],
            axis=1)
        return 0.5 * norm, gA

    def vertex_areas(self, X):
        A, _ = self.tri_geom(X)
        Av = np.zeros(len(X))
        np.add.at(Av, self.faces.ravel(), np.repeat(A / 3.0, 3))
        return Av

    def min_edge(self, X):
        F = self.faces
        e = [X[F[:, i]] - X[F[:, j]] for i, j in ((0, 1), (1, 2), (2, 0))]
        return min(np.linalg.norm(v, axis=1).min() for v in e)

    # -- dynamics ---------------------------------------------------------
    def velocity(self, X, p):
        m = self.sigma * self.vertex_areas(X)
        return p / np.sqrt(m ** 2 + np.einsum('ij,ij->i', p, p))[:, None]

    def rhs(self, X, p):
        V = self.velocity(X, p)
        A, gA = self.tri_geom(X)
        root = np.sqrt(np.clip(1.0 - np.einsum('ij,ij->i', V, V), 1e-14, None))
        sT = root[self.faces].mean(axis=1)
        Fv = np.zeros_like(X)
        for k in range(3):
            np.add.at(Fv, self.faces[:, k], -self.sigma * sT[:, None] * gA[:, k])
        return V, Fv

    def energy(self):
        m = self.sigma * self.vertex_areas(self.X)
        return np.sum(np.sqrt(m ** 2 + np.einsum('ij,ij->i', self.p, self.p)))

    def vmax(self):
        V = self.velocity(self.X, self.p)
        return np.sqrt(np.einsum('ij,ij->i', V, V).max())

    def tangential_smooth(self, lam=0.5, conserve_energy=True):
        """Re-gauge the mesh: tangential redistribution of vertices.

        Tangential motion is pure gauge for a Nambu-Goto wall, so this changes
        the discretization, not the physics. Steps: (1) tangential part of the
        uniform-weight umbrella Laplacian, (2) reproject onto the old mesh
        (closest point over the 2-ring), (3) barycentric re-interpolation of
        velocity at the new material point, (4) rebuild momenta. Returns
        (relative energy change, relative momentum change) for monitoring.

        Linear interpolation of u systematically under-counts the energy
        E = sum m sqrt(1+u^2) (convexity in u), ~1e-4 relative per pass, which
        dominates the total drift. Since the pass is gauge, conserve_energy
        restores E exactly afterwards by a single global rescale u -> s*u
        (Newton on s^2; monotonic, so 3-4 iterations reach machine precision).
        The returned energy change is the raw pre-rescale loss, so monitoring
        still sees what interpolation cost before correction.
        """
        X0, p0 = self.X, self.p
        E0 = self.energy()
        P0 = p0.sum(axis=0)
        V0 = self.velocity(X0, p0)

        # vertex normals: area-weighted average of incident face normals
        A, _ = self.tri_geom(X0)
        a, b, c = (X0[self.faces[:, k]] for k in range(3))
        fn = np.cross(b - a, c - a)  # |fn| = 2A, so summing is area weighting
        vn = np.zeros_like(X0)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1)[:, None]

        # tangential umbrella displacement
        nsum = np.add.reduceat(X0[self.nbr_flat], self.nbr_off[:-1])
        cnt = np.diff(self.nbr_off)
        delta = nsum / cnt[:, None] - X0
        delta -= np.einsum('ij,ij->i', delta, vn)[:, None] * vn
        Xnew = X0 + lam * delta

        # reproject onto old mesh: closest point among 2-ring candidates
        P = Xnew[self.cand_v]
        TA, TB, TC = (X0[self.faces[self.cand_t, k]] for k in range(3))
        bary = closest_point_bary(P, TA, TB, TC)
        cpts = bary[:, 0:1] * TA + bary[:, 1:2] * TB + bary[:, 2:3] * TC
        d2 = np.einsum('ij,ij->i', P - cpts, P - cpts)
        best = np.full(len(X0), np.inf)
        np.minimum.at(best, self.cand_v, d2)
        take = np.zeros(len(X0), dtype=int)
        hit = d2 <= best[self.cand_v] * (1 + 1e-12)
        take[self.cand_v[hit]] = np.nonzero(hit)[0]
        self.X = cpts[take]

        # re-interpolate the proper velocity u = gamma*V: u is unbounded and
        # smooth where V saturates at 1, so interpolating it avoids the
        # 1/(1-v^2) energy-error amplification; |V| = |u|/sqrt(1+u^2) < 1
        # stays automatic, and p = sigma*Atil*gamma*V = sigma*Atil*u exactly.
        gam0 = 1.0 / np.sqrt(np.clip(1 - np.einsum('ij,ij->i', V0, V0),
                                     1e-14, None))
        u0 = gam0[:, None] * V0
        ub = u0[self.faces[self.cand_t[take]]]  # (n, corner, 3)
        unew = np.einsum('nc,ncj->nj', bary[take], ub)

        m = self.sigma * self.vertex_areas(self.X)
        self.p = m[:, None] * unew
        E1 = self.energy()

        if conserve_energy and E0 > m.sum():  # no root exists below rest energy
            u2 = np.einsum('ij,ij->i', unew, unew)
            w = 1.0
            for _ in range(6):
                root = np.sqrt(1.0 + w * u2)
                g = np.sum(m * root) - E0
                gp = 0.5 * np.sum(m * u2 / root)
                if gp <= 0:  # static wall: rescaling u cannot restore E
                    w = 1.0
                    break
                w -= g / gp
                if not np.isfinite(w) or w <= 0:
                    w = 1.0  # Newton overshot out of the domain: skip the
                    break    # rescale this pass rather than poison p with NaN
            self.p *= np.sqrt(w)

        P1 = self.p.sum(axis=0)
        return abs(E1 / E0 - 1.0), np.linalg.norm(P1 - P0) / (E0 + 1e-300)

    def collapse_short_edges(self, lmin, max_pass=10):
        """Coarsen the mesh: collapse every edge shorter than lmin.

        Each collapse merges the edge's vertices into one at their
        energy-weighted midpoint (the center of energy of the pair) with
        p_new = p_a + p_b, so total momentum is conserved exactly. Energy
        changes (removing sub-grid structure is dissipation, as in surgery)
        and is reported, not rescaled. Guards, standard for manifold meshes:
        the link condition (the two rings share exactly the two opposite
        vertices) and a normal-flip test on the surviving incident faces;
        candidates failing either are skipped. Accepted collapses in one pass
        are kept independent by freezing both 1-rings. Returns
        (n_collapsed, |dE|/E, |dP|/E).
        """
        E0 = self.energy()
        P0 = self.p.sum(axis=0)
        total = 0
        for _ in range(max_pass):
            n = len(self.X)
            F = self.faces
            edges = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]],
                                       F[:, [2, 0]]]), axis=1)
            edges = np.unique(edges, axis=0)
            el = np.linalg.norm(self.X[edges[:, 0]] - self.X[edges[:, 1]],
                                axis=1)
            order = np.argsort(el)
            order = order[el[order] < lmin]
            if len(order) == 0:
                break
            nbr = [set() for _ in range(n)]
            vtri = [set() for _ in range(n)]
            for ti, (a, b, c) in enumerate(F):
                nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
                vtri[a].add(ti); vtri[b].add(ti); vtri[c].add(ti)
            m = self.sigma * self.vertex_areas(self.X)
            Ev = np.sqrt(m ** 2 + np.einsum('ij,ij->i', self.p, self.p))
            frozen = np.zeros(n, bool)
            pairs = []
            for i in order:
                a, b = edges[i]
                if frozen[a] or frozen[b]:
                    continue
                if len(nbr[a] & nbr[b]) != 2:      # link condition
                    continue
                # normal-flip test on surviving faces of the joint 1-ring
                mid = (Ev[a] * self.X[a] + Ev[b] * self.X[b]) / (Ev[a] + Ev[b])
                flip = False
                for ti in (vtri[a] | vtri[b]):
                    tri = F[ti]
                    if a in tri and b in tri:
                        continue                    # face dies with the edge
                    P = self.X[tri].copy()
                    n_old = np.cross(P[1] - P[0], P[2] - P[0])
                    P[tri == a] = mid
                    P[tri == b] = mid
                    n_new = np.cross(P[1] - P[0], P[2] - P[0])
                    denom = np.linalg.norm(n_old) * np.linalg.norm(n_new)
                    if denom < 1e-300 or n_old @ n_new < 0.2 * denom:
                        flip = True
                        break
                if flip:
                    continue
                pairs.append((a, b, mid))
                for v in {a, b} | nbr[a] | nbr[b]:
                    frozen[v] = True
            if not pairs:
                break
            remap = np.arange(n)
            dead = np.zeros(n, bool)
            for a, b, mid in pairs:
                self.X[a] = mid
                self.p[a] = self.p[a] + self.p[b]
                dead[b] = True
                remap[b] = a
            F2 = remap[F]
            good = ((F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2])
                    & (F2[:, 2] != F2[:, 0]))
            newid = np.full(n, -1)
            newid[~dead] = np.arange((~dead).sum())
            self.X = self.X[~dead]
            self.p = self.p[~dead]
            self.faces = newid[F2[good]]
            total += len(pairs)
            self._build_adjacency()
        if total == 0:
            return 0, 0.0, 0.0
        E1 = self.energy()
        P1 = self.p.sum(axis=0)
        return total, abs(E1 / E0 - 1.0), np.linalg.norm(P1 - P0) / E0

    def step(self, dt):
        X0, p0 = self.X, self.p
        k1x, k1p = self.rhs(X0, p0)
        k2x, k2p = self.rhs(X0 + 0.5 * dt * k1x, p0 + 0.5 * dt * k1p)
        k3x, k3p = self.rhs(X0 + 0.5 * dt * k2x, p0 + 0.5 * dt * k2p)
        k4x, k4p = self.rhs(X0 + dt * k3x, p0 + dt * k3p)
        self.X = X0 + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        self.p = p0 + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        self.t += dt


def closest_point_bary(P, A, B, C):
    """Barycentric coords of the closest point to P on triangle ABC.

    Vectorized version of Ericson, 'Real-Time Collision Detection', 5.1.5.
    All inputs (n,3); returns (n,3) barycentric weights (u,v,w) summing to 1.
    """
    ab, ac = B - A, C - A
    ap = P - A
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = P - B
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = P - C
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = 1e-300
    # interior (default), then overwrite edge/vertex regions
    denom = va + vb + vc
    v_in = vb / np.where(np.abs(denom) < eps, eps, denom)
    w_in = vc / np.where(np.abs(denom) < eps, eps, denom)
    u = 1.0 - v_in - w_in
    bary = np.stack([u, v_in, w_in], axis=1)

    def put(mask, u, v, w):
        bary[mask] = np.stack([u, v, w], axis=1)[mask]

    t_bc = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < eps, eps,
                                (d4 - d3) + (d5 - d6))
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
        np.zeros_like(t_bc), 1 - t_bc, t_bc)
    t_ac = d2 / np.where(np.abs(d2 - d6) < eps, eps, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t_ac, np.zeros_like(t_ac), t_ac)
    t_ab = d1 / np.where(np.abs(d1 - d3) < eps, eps, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t_ab, t_ab, np.zeros_like(t_ab))
    zero, one = np.zeros(len(P)), np.ones(len(P))
    put((d6 >= 0) & (d5 <= d6), zero, zero, one)
    put((d3 >= 0) & (d4 <= d3), zero, one, zero)
    put((d1 <= 0) & (d2 <= 0), one, zero, zero)
    return bary


def _selftest_grad_area():
    """Finite-difference check of the area gradient."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 3))
    mem = Membrane(X, [[0, 1, 2]])
    A, gA = mem.tri_geom(X)
    eps = 1e-6
    for corner in range(3):
        for k in range(3):
            Xp = X.copy(); Xp[corner, k] += eps
            Xm = X.copy(); Xm[corner, k] -= eps
            fd = (mem.tri_geom(Xp)[0][0] - mem.tri_geom(Xm)[0][0]) / (2 * eps)
            assert abs(fd - gA[0, corner, k]) < 1e-7, (corner, k, fd, gA[0, corner, k])


def _selftest_closest_point():
    """closest_point_bary vs brute-force sampling of the triangle."""
    rng = np.random.default_rng(1)
    A, B, C = rng.normal(size=(3, 40, 3))
    P = rng.normal(size=(40, 3)) * 2
    bary = closest_point_bary(P, A, B, C)
    assert np.allclose(bary.sum(axis=1), 1, atol=1e-12)
    assert (bary > -1e-12).all()
    cp = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    d = np.linalg.norm(P - cp, axis=1)
    u = np.linspace(0, 1, 120)
    uu, vv = np.meshgrid(u, u)
    m = uu + vv <= 1.0
    uu, vv = uu[m], vv[m]
    for i in range(len(P)):
        grid = (1 - uu - vv)[:, None] * A[i] + uu[:, None] * B[i] + vv[:, None] * C[i]
        dbrute = np.linalg.norm(grid - P[i], axis=1).min()
        assert d[i] <= dbrute + 1e-3, (i, d[i], dbrute)


def _selftest_collapse():
    """Edge collapse must keep a closed manifold and conserve momentum."""
    V, F = icosphere(2)
    mem = Membrane(V, F)
    m = mem.sigma * mem.vertex_areas(V)
    mem.p = 0.5 * m[:, None] * (-V)                 # infalling sphere
    P0 = mem.p.sum(axis=0)
    n0 = len(mem.X)
    nc, dE, dP = mem.collapse_short_edges(1.2 * mem.min_edge(mem.X))
    assert nc > 0 and len(mem.X) == n0 - nc
    assert dP < 1e-12
    F2 = mem.faces
    edges = np.sort(np.vstack([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]]),
                    axis=1)
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    assert (cnt == 2).all()                          # closed manifold
    chi = len(mem.X) - len(uniq) + len(F2)
    assert chi == 2, chi
    assert np.isfinite(mem.energy())


_selftest_grad_area()
_selftest_closest_point()
_selftest_collapse()
