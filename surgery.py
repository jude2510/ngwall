"""Intercommutation surgery for a pinching Nambu-Goto wall.

When a closed wall's surface self-contacts at a neck (detected here as a ring
of vertices approaching the symmetry axis, rho -> 0), the wall reconnects:
the throat is excised and the two exposed boundary loops are capped, splitting
the mesh into two closed daughter walls that then evolve independently.

Energy bookkeeping is report-only by design: unlike tangential smoothing,
surgery is not pure gauge -- the excised throat is the analog of the string
"bridge" and its energy is physically lost (radiated / annihilated). The
returned report carries E_before, E_removed, the daughters' energies and the
net dE, dP so that the loss can be studied as an observable. Vertex proper
velocities u = gamma*V are preserved through the split and momenta rebuilt
from the new vertex areas, p = sigma*Atil*u (interior vertices are untouched
since their areas do not change).

Caps introduce no new vertices: each boundary loop is closed by its
minimum-area triangulation (DP, O(n^3)). At the pinch the throat mesh is
sub-grid crumple, so the cut boundary is a strongly zigzagging space curve
(perimeter >> 2*pi*rho); a centroid fan over such a loop sweeps a large cone
and inserts spurious area*gamma energy at the seam, while the minimum-area
stitch bounds the inserted area from below among triangulations.
"""

import numpy as np
from membrane import Membrane


def detect_pinch(mem, mask, rho_c=0.02):
    """Return the pinch height z* if min rho over masked vertices < rho_c."""
    rho = np.hypot(mem.X[:, 0], mem.X[:, 1])
    a = np.nonzero(mask)[0][np.argmin(rho[mask])]
    return mem.X[a, 2] if rho[a] < rho_c else None


def _boundary_loops(faces):
    """Directed boundary edges of an oriented triangle mesh -> vertex loops.

    Interior edges appear twice in opposite directions; boundary edges once.
    Each loop is returned in the boundary-edge direction. Raises if the
    boundary is not a disjoint union of simple cycles.
    """
    edges = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if (v, u) in edges:
                del edges[(v, u)]
            else:
                edges[(u, v)] = None
    succ = {}
    for u, v in edges:
        if u in succ:
            raise ValueError(f"non-manifold boundary at vertex {u}")
        succ[u] = v
    loops = []
    while succ:
        start, v = next(iter(succ.items()))
        loop = [start]
        u = succ.pop(start)
        while u != start:
            loop.append(u)
            u = succ.pop(u)
        loops.append(loop)
    return loops


def _cap_tri_forbidden(verts, i, k, j, n, bad_faces, bad_edges):
    """True if cap triangle (i,k,j) would degenerate the complex: it
    duplicates an existing face, or one of its DIAGONALS (non-boundary
    polygon chords) duplicates an existing interior mesh edge."""
    if bad_faces is None:
        return False
    a, b, c = verts[i], verts[k], verts[j]
    if frozenset((a, b, c)) in bad_faces:
        return True
    if k != i + 1 and frozenset((a, b)) in bad_edges:
        return True
    if j != k + 1 and frozenset((b, c)) in bad_edges:
        return True
    if not (i == 0 and j == n - 1) and frozenset((a, c)) in bad_edges:
        return True
    return False


def _min_area_cap(P, verts=None, bad_faces=None, bad_edges=None):
    """Minimum-area valid triangulation of a closed 3D polygon (DP, O(n^3)).

    P (n,3) is the vertex cycle. Returns index triples (i, k, j), i < k < j,
    whose triangles traverse every polygon edge in the forward direction and
    every internal diagonal twice in opposite directions -- so to cap an
    oriented mesh boundary, pass the loop REVERSED and the cap winds against
    the surviving boundary edges, closing the surface consistently.

    With verts/bad_faces/bad_edges given (vertex ids of the loop, existing
    faces as frozensets, existing interior edges as frozensets), triangles
    that would duplicate existing geometry cost infinity: the result is the
    minimum-area triangulation that keeps the complex a manifold. Returns
    None if no valid triangulation exists (caller should dilate).
    """
    n = len(P)
    cost = np.zeros((n, n))
    split = np.zeros((n, n), dtype=int)
    for span in range(2, n):
        for i in range(n - span):
            j = i + span
            k = np.arange(i + 1, j)
            tri = 0.5 * np.linalg.norm(
                np.cross(P[k] - P[i], P[j] - P[i]), axis=1)
            if bad_faces is not None:
                pen = np.array([_cap_tri_forbidden(verts, i, kk, j, n,
                                                   bad_faces, bad_edges)
                                for kk in k])
                tri = np.where(pen, np.inf, tri)
            tot = cost[i, k] + cost[k, j] + tri
            b = np.argmin(tot)
            cost[i, j] = tot[b]
            split[i, j] = k[b]
    if not np.isfinite(cost[0, n - 1]):
        return None
    tris, stack = [], [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = split[i, j]
        tris.append((i, k, j))
        stack += [(i, k), (k, j)]
    return tris


def _components(nvert, faces):
    """Connected-component label per vertex (-1 for vertices with no face)."""
    label = np.full(nvert, -1)
    adj = [[] for _ in range(nvert)]
    for a, b, c in faces:
        adj[a] += [b, c]; adj[b] += [a, c]; adj[c] += [a, b]
    comp = 0
    for seed in faces[:, 0]:
        if label[seed] != -1:
            continue
        stack = [seed]
        label[seed] = comp
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if label[v] == -1:
                    label[v] = comp
                    stack.append(v)
        comp += 1
    return label, comp


def pinch_split(mem, z_star, r_cut=0.04, h=0.03):
    """Excise the throat at z*, cap the two loops, split into daughters.

    Returns (daughters, report): a list of Membrane objects and a dict with
    the surgery energy/momentum bookkeeping.
    """
    X, p, F = mem.X, mem.p, mem.faces
    rho = np.hypot(X[:, 0], X[:, 1])
    S = (rho < r_cut) & (np.abs(X[:, 2] - z_star) < h)
    if not S.any():
        raise ValueError("empty throat band; increase r_cut/h")
    keep = ~S[F].any(axis=1)
    F2 = F[keep]

    loops = _boundary_loops(F2)
    if len(loops) != 2:
        raise ValueError(f"expected 2 boundary loops, found {len(loops)}")

    E_before = mem.energy()
    P_before = p.sum(axis=0)
    mE = mem.sigma * mem.vertex_areas(X)
    Ev = np.sqrt(mE ** 2 + np.einsum('ij,ij->i', p, p))
    E_removed = Ev[S].sum()
    V = mem.velocity(X, p)
    gam = 1.0 / np.sqrt(np.clip(1 - np.einsum('ij,ij->i', V, V), 1e-14, None))
    u = gam[:, None] * V

    # cap each loop with its minimum-area triangulation (no new vertices);
    # feed the loop reversed so the cap winds against the boundary direction
    Xa, ua = X, u
    newF = list(F2)
    for loop in loops:
        R = np.array(loop[::-1])
        for i, k, j in _min_area_cap(X[R]):
            newF.append([R[i], R[k], R[j]])
    newF = np.array(newF)

    label, ncomp = _components(len(Xa), newF)
    if ncomp != 2:
        raise ValueError(f"expected 2 daughters, found {ncomp}")

    daughters, reports = [], []
    for cid in range(ncomp):
        vids = np.nonzero(label == cid)[0]
        remap = np.full(len(Xa), -1)
        remap[vids] = np.arange(len(vids))
        fsub = remap[newF[(label[newF[:, 0]] == cid)]]
        d = Membrane(Xa[vids], fsub, sigma=mem.sigma)
        d.t = mem.t
        md = d.sigma * d.vertex_areas(d.X)
        d.p = md[:, None] * ua[vids]
        chi = len(vids) - _edge_count(fsub) + len(fsub)
        if chi != 2:
            raise ValueError(f"daughter {cid} not a closed surface (chi={chi})")
        daughters.append(d)
        reports.append(d.energy())

    report = dict(E_before=E_before, E_removed=E_removed,
                  E_daughters=reports, dE=sum(reports) - E_before,
                  dP=np.linalg.norm(
                      sum(d.p.sum(axis=0) for d in daughters) - P_before),
                  n_removed=int(S.sum()),
                  loop_sizes=[len(l) for l in loops])
    return daughters, report


def _edge_count(faces):
    e = set()
    for a, b, c in faces:
        e |= {frozenset((a, b)), frozenset((b, c)), frozenset((c, a))}
    return len(e)


def _nonmanifold_cut_verts(faces):
    """Vertices where the cut boundary crosses itself (more than one
    boundary edge in or out) -- the targets for targeted dilation when
    _boundary_loops refuses the mesh."""
    from collections import Counter
    edges = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if (v, u) in edges:
                del edges[(v, u)]
            else:
                edges[(u, v)] = None
    outc = Counter(u for u, v in edges)
    inc = Counter(v for u, v in edges)
    return sorted({u for u, c in outc.items() if c > 1}
                  | {v for v, c in inc.items() if c > 1})


def _vertex_pinches(faces, loops):
    """Vertices that would pinch the stitched surface into a point-join.

    Two kinds: a vertex whose incident surviving faces do not form a single
    edge-connected fan, and a vertex shared by two different boundary loops.
    Returns the offending vertex ids (empty list = vertex-manifold).
    """
    from collections import defaultdict
    bad = set()
    vf = defaultdict(list)
    for ti, tri in enumerate(faces):
        for v in tri:
            vf[v].append(ti)
    for v, tis in vf.items():
        wing = defaultdict(list)          # neighbor vertex -> faces at v
        for ti in tis:
            for w in faces[ti]:
                if w != v:
                    wing[w].append(ti)
        parent = {ti: ti for ti in tis}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for fs in wing.values():
            for f in fs[1:]:
                parent[find(f)] = find(fs[0])
        if len({find(ti) for ti in tis}) > 1:
            bad.add(v)
    seen = {}
    for li, loop in enumerate(loops):
        for v in loop:
            if v in seen and seen[v] != li:
                bad.add(v)
            seen[v] = li
    return sorted(bad)


def _assembly_problem(faces):
    """'' if the stitched mesh is clean, else a short description.

    Clean: no duplicate faces, every undirected edge in exactly two faces
    (closed manifold), and every connected component has an even Euler
    characteristic <= 2.
    """
    from collections import Counter
    dup = Counter(frozenset(t) for t in faces.tolist())
    ndup = sum(v - 1 for v in dup.values() if v > 1)
    if ndup:
        return f"{ndup} duplicate faces"
    und = Counter(frozenset((t[i], t[(i + 1) % 3]))
                  for t in faces.tolist() for i in range(3))
    bad = [v for v in und.values() if v != 2]
    if bad:
        return f"{len(bad)} edges with multiplicity != 2"
    label, ncomp = _components(int(faces.max()) + 1, faces)
    for cid in range(ncomp):
        fsub = faces[label[faces[:, 0]] == cid]
        vids = np.unique(fsub)
        chi = len(vids) - _edge_count(fsub) + len(fsub)
        if chi > 2 or chi % 2:
            return f"component {cid} has chi = {chi}"
    return ""


def _tri_components(faces):
    """Number of connected components of a triangle set (shared vertices)."""
    if len(faces) == 0:
        return 0
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in faces:
        for v in tri:
            parent.setdefault(v, v)
        r = find(tri[0])
        for v in tri[1:]:
            parent[find(v)] = r
    return len({find(v) for v in parent})


def _min_area_tube(PA, PB, VA=None, VB=None, bad_faces=None, bad_edges=None):
    """Minimum-area valid triangulated annulus joining polygons PA, PB.

    Standard contour-stitching DP over the (i, j) staircase, run for every
    starting alignment of B (O(n m^2)). Returns triangles as (side, i, side,
    j, side, k) index triples via lists tri_a-advance (i, i+1, j) and
    tri_b-advance (i, j, j+1), with both polygons traversed FORWARD -- so
    pass side A reversed (its edges must oppose the surviving boundary) and
    side B as returned by _boundary_loops (a cylinder traverses B backward).

    With VA/VB (vertex ids) and bad_faces/bad_edges given, tube triangles
    duplicating existing faces or whose cross edges duplicate existing
    interior mesh edges cost infinity; returns None if no valid stitching
    exists (caller should dilate).
    """
    n, m = len(PA), len(PB)

    def tri_area(p, q, r):
        return 0.5 * np.linalg.norm(np.cross(q - p, r - p))

    def bad_a(ia, jb):      # triangle (A_ia, A_ia+1, B_jb), mapped indices
        if bad_faces is None:
            return False
        a, b, c = VA[ia % n], VA[(ia + 1) % n], VB[jb]
        return (frozenset((a, b, c)) in bad_faces
                or frozenset((a, c)) in bad_edges
                or frozenset((b, c)) in bad_edges)

    def bad_b(ia, jb1, jb0):    # triangle (A_ia, B_jb1, B_jb0), mapped
        if bad_faces is None:
            return False
        a, b, c = VA[ia % n], VB[jb1], VB[jb0]
        return (frozenset((a, b, c)) in bad_faces
                or frozenset((a, b)) in bad_edges
                or frozenset((a, c)) in bad_edges)

    best = (np.inf, None, None)
    for j0 in range(m):
        B = np.roll(np.arange(m), -j0)
        cost = np.full((n + 1, m + 1), np.inf)
        move = np.zeros((n + 1, m + 1), np.int8)
        cost[0, 0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if i < n:
                    if bad_a(i, B[j % m]):
                        c = np.inf
                    else:
                        c = cost[i, j] + tri_area(PA[i % n], PA[(i + 1) % n],
                                                  PB[B[j % m]])
                    if c < cost[i + 1, j]:
                        cost[i + 1, j] = c
                        move[i + 1, j] = 1          # came by advancing A
                if j < m:
                    if bad_b(i, B[(j + 1) % m], B[j % m]):
                        c = np.inf
                    else:
                        c = cost[i, j] + tri_area(PA[i % n],
                                                  PB[B[(j + 1) % m]],
                                                  PB[B[j % m]])
                    if c < cost[i, j + 1]:
                        cost[i, j + 1] = c
                        move[i, j + 1] = 2          # came by advancing B
        if cost[n, m] < best[0]:
            best = (cost[n, m], move.copy(), B.copy())
    if best[1] is None or not np.isfinite(best[0]):
        return None
    _, move, B = best
    tris, i, j = [], n, m
    while i > 0 or j > 0:
        if move[i, j] == 1:
            tris.append(("a", (i - 1) % n, i % n, B[j % m]))
            i -= 1
        else:
            tris.append(("b", i % n, (j) % m, (j - 1) % m))
            j -= 1
    out = []
    for t in tris:
        if t[0] == "a":
            _, ia, ib, jb = t
            out.append(("aab", ia, ib, jb))
        else:
            _, ia, jb1, jb0 = t
            out.append(("abb", ia, B[jb1 % m], B[jb0 % m]))
    return out


def contact_surgery(mem, cluster, max_dilate=8, min_faces=4, d_c=None,
                    debris_extent=None, pair_dist=None):
    """Reconnect the wall at one contact cluster (vertex-index array).

    The rule, loop by loop: excise all triangles touching the (dilated)
    cluster; drop surviving components smaller than debris_extent as
    annihilated overlap debris (walls overlapping within the resolution
    scale are locally wall-antiwall and annihilate); then pair the remaining
    boundary loops -- two loops whose mean surface normals FACE each other
    (dot < -0.5) within pair_dist are tunnel-stitched (reconnection), and
    every unpaired loop is capped (split). Throat loops bulge outward, never
    face, and so cap; facing sheets tunnel on first touch; renewed contact
    around an existing tunnel strands the old sleeve as debris and
    re-tunnels the outer loops -- the hole WIDENS instead of adding handles.

    A shredded excision (non-manifold cut, pinched vertices, or a stitched
    candidate that fails assembly checks) dilates the flagged region by one
    mesh ring and retries, up to max_dilate times. If excision consumes
    every kept face the wall has annihilated (walls = [], mode
    "annihilated"). Proper velocity u is preserved and momenta rebuilt from
    new areas; dE is reported, not rescaled. Returns (walls, report).
    """
    X, p, F = mem.X, mem.p, mem.faces
    if d_c is None:
        d_c = 0.2 * np.median(
            np.linalg.norm(X[F[:, 0]] - X[F[:, 1]], axis=1))
    debris_extent = debris_extent if debris_extent is not None else 3 * d_c
    pair_dist = pair_dist if pair_dist is not None else 4 * d_c
    cluster0 = np.asarray(cluster)
    S = np.zeros(len(X), bool)
    S[cluster0] = True
    if not S[F].any():
        raise ValueError("empty excision for contact cluster")
    nbr = [set() for _ in range(len(X))]
    for a, b, c in F:
        nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
    mE = mem.sigma * mem.vertex_areas(X)
    Ev = np.sqrt(mE ** 2 + np.einsum('ij,ij->i', p, p))

    annihilated = False
    newF = None
    why = ""
    E_debris = 0.0
    n_tunnel = n_cap = 0
    loops = []
    for it in range(max_dilate + 1):
        kill = S[F].any(axis=1)
        F2 = F[~kill]
        E_debris = 0.0
        if len(F2) == 0:
            annihilated = True
            break
        # drop sub-resolution surviving components: annihilated overlap
        labF, ncF = _components(len(X), F2)
        keep = []
        for cid in range(ncF):
            fm = labF[F2[:, 0]] == cid
            vids = np.unique(F2[fm])
            ext = (X[vids].max(axis=0) - X[vids].min(axis=0)).max()
            if fm.sum() < min_faces or ext < debris_extent:
                E_debris += Ev[vids].sum()
            else:
                keep.append(F2[fm])
        if not keep:
            annihilated = True
            break
        F2k = np.concatenate(keep)
        try:
            loops = _boundary_loops(F2k)
            nmv = []
        except ValueError:
            loops = None
            nmv = _nonmanifold_cut_verts(F2k)
        pinch = _vertex_pinches(F2k, loops) if loops is not None else []
        if loops is not None and not pinch:
            # loop mean normals (from surviving faces) and centroids
            a, b, c = X[F2k[:, 0]], X[F2k[:, 1]], X[F2k[:, 2]]
            fn = np.cross(b - a, c - a)
            vn = np.zeros_like(X)
            for k3 in range(3):
                np.add.at(vn, F2k[:, k3], fn)
            nbar, cent = [], []
            for l in loops:
                s = vn[l].sum(axis=0)
                nbar.append(s / max(np.linalg.norm(s), 1e-300))
                cent.append(X[l].mean(axis=0))
            cand_pairs = sorted(
                (np.linalg.norm(cent[i] - cent[j]), i, j)
                for i in range(len(loops)) for j in range(i + 1, len(loops))
                if nbar[i] @ nbar[j] < -0.5
                and np.linalg.norm(cent[i] - cent[j]) < pair_dist)
            used, pairs = set(), []
            for _, i, j in cand_pairs:
                if i not in used and j not in used:
                    pairs.append((i, j))
                    used |= {i, j}
            unpaired = [i for i in range(len(loops)) if i not in used]
            # existing geometry is off-limits to the stitches: a min-area
            # cap over a fold otherwise reproduces the fold's own triangles
            from collections import Counter
            ecnt = Counter(frozenset((t[a0], t[(a0 + 1) % 3]))
                           for t in F2k.tolist() for a0 in range(3))
            bad_edges = {ee for ee, cc in ecnt.items() if cc == 2}
            bad_faces = {frozenset(t) for t in F2k.tolist()}
            cand = list(F2k)
            feasible = True
            for i, j in pairs:
                A = np.array(loops[i][::-1])
                B = np.array(loops[j])
                tube = _min_area_tube(X[A], X[B], A, B, bad_faces, bad_edges)
                if tube is None:
                    feasible = False
                    break
                for kind, i0, i1, jj in tube:
                    cand.append([A[i0], A[i1], B[jj]] if kind == "aab"
                                else [A[i0], B[i1], B[jj]])
            if feasible:
                for i in unpaired:
                    R = np.array(loops[i][::-1])
                    capt = _min_area_cap(X[R], R, bad_faces, bad_edges)
                    if capt is None:
                        feasible = False
                        break
                    for i0, k, j in capt:
                        cand.append([R[i0], R[k], R[j]])
            if feasible:
                cand = np.array(cand)
                why = _assembly_problem(cand)
                if not why:
                    newF = cand
                    n_tunnel, n_cap = len(pairs), len(unpaired)
                    break
            else:
                why = "no valid triangulation at resolution"
        else:
            why = (f"{'non-manifold cut' if loops is None else ''}"
                   f"{len(pinch) if loops is not None else ''}"
                   f"{' pinched vertices' if loops is not None else ''}")
        if it == max_dilate:
            raise ValueError(f"contact surgery unresolved after "
                             f"{max_dilate} dilations: {why}")
        if it >= 4:
            # graph dilation crawls along one sheet, but late-cascade crumple
            # zones are multi-layered: several folded sheets within d_c in
            # SPACE yet far apart on the mesh. A Euclidean ball around the
            # original cluster takes every layer in one bite -- "this whole
            # neighborhood is in contact at resolution scale".
            from scipy.spatial import cKDTree
            r = (it - 2) * d_c                 # 2*d_c, 3*d_c, 4*d_c, ...
            hit = set()
            for ids in cKDTree(X).query_ball_point(X[cluster0], r):
                hit.update(ids)
            S[:] = False
            S[list(hit)] = True
        elif loops is not None and pinch:      # targeted dilation first
            S[pinch] = True
        elif loops is None and nmv:            # cut-crossing vertices too
            S[nmv] = True
        else:
            grow = set()
            for v in np.nonzero(S)[0]:
                grow |= nbr[v]
            S[list(grow)] = True
    mode = ("tunnel" if n_tunnel and not n_cap else
            "cap-split" if n_cap and not n_tunnel else
            "tunnel+cap" if n_tunnel else "excise")

    E_before = mem.energy()
    P_before = p.sum(axis=0)
    E_removed = Ev[S].sum()
    if annihilated:
        return [], dict(mode="annihilated", E_before=E_before,
                        E_removed=E_removed, E_walls=[], E_debris=E_debris,
                        dE=-E_before, dP=np.linalg.norm(P_before),
                        n_removed=int(S.sum()), loop_sizes=[], genus=[],
                        n_tunnel=0, n_cap=0)
    V = mem.velocity(X, p)
    gam = 1.0 / np.sqrt(np.clip(1 - np.einsum('ij,ij->i', V, V), 1e-14, None))
    u = gam[:, None] * V

    label, ncomp = _components(len(X), newF)
    walls, energies = [], []
    for cid in range(ncomp):
        vids = np.nonzero(label == cid)[0]
        remap = np.full(len(X), -1)
        remap[vids] = np.arange(len(vids))
        fsub = remap[newF[(label[newF[:, 0]] == cid)]]
        if len(fsub) < min_faces:              # sub-resolution scrap
            E_debris += Ev[vids].sum()
            continue
        if len(_boundary_loops(fsub)) != 0:
            raise ValueError(f"wall {cid} not closed after {mode}")
        chi = len(vids) - _edge_count(fsub) + len(fsub)
        if chi > 2 or chi % 2:
            raise ValueError(f"wall {cid} bad Euler characteristic {chi}")
        d = Membrane(X[vids], fsub, sigma=mem.sigma)
        d.t = mem.t
        md = d.sigma * d.vertex_areas(d.X)
        d.p = md[:, None] * u[vids]
        d.genus = (2 - chi) // 2
        walls.append(d)
        energies.append(d.energy())

    report = dict(mode=mode, E_before=E_before, E_removed=E_removed,
                  E_walls=energies, E_debris=E_debris,
                  dE=sum(energies) - E_before,
                  dP=np.linalg.norm(
                      sum(w.p.sum(axis=0) for w in walls) - P_before)
                  if walls else np.linalg.norm(P_before),
                  n_removed=int(S.sum()),
                  loop_sizes=[len(l) for l in loops],
                  genus=[w.genus for w in walls],
                  n_tunnel=n_tunnel, n_cap=n_cap)
    return walls, report


def _selftest_split():
    """Topology check: splitting an icosphere at its equator gives 2 caps."""
    from membrane import icosphere
    X, F = icosphere(3)
    mem = Membrane(X, F)
    mem.p = 0.3 * mem.sigma * mem.vertex_areas(X)[:, None] * (-X)  # infalling
    # equator band plays the role of the throat: rho there is ~1, so widen
    daughters, rep = pinch_split(mem, z_star=0.0, r_cut=2.0, h=0.15)
    assert len(daughters) == 2
    assert rep["dP"] < 1e-12 + abs(rep["dE"]) * 10  # loose: report sanity
    for d in daughters:
        assert len(_boundary_loops(d.faces)) == 0  # closed
        z = d.X[:, 2]
        assert (z > -0.2).all() or (z < 0.2).all()  # hemispheres


_selftest_split()
