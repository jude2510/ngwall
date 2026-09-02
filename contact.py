"""Generic self-contact detection for closed wall meshes.

A contact is a pair of vertices that are (1) within the contact distance d_c
in space, (2) far apart ON the surface (graph distance > k_ring, so adjacent
mesh never self-flags), and (3) on opposing sheets (vertex normals with
dot < normal_dot). Flagged pairs are clustered into contact regions by
connected components over the union of contact pairs and mesh edges between
flagged vertices; each cluster is one surgery site for
surgery.contact_surgery, whose excised-set topology decides cap vs tunnel.

d_c should sit at the resolution floor: with coarsening at lmin, local edges
never shrink below ~lmin, so d_c = 2*lmin means "sheets closer than the mesh
can resolve" -- the wall analog of the string intercommutation condition.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def vertex_normals(X, faces):
    """Area-weighted outward vertex normals (unit)."""
    a, b, c = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    vn = np.zeros_like(X)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.clip(np.linalg.norm(vn, axis=1), 1e-300, None)[:, None]


def _khop(faces, n, k):
    """Boolean CSR reachability matrix for graph distance <= k."""
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                        faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                        faces[:, 0], faces[:, 1], faces[:, 2]])
    A = sparse.csr_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n))
    A = (A + sparse.identity(n, dtype=bool, format="csr")).astype(bool)
    R = A.copy()
    for _ in range(k - 1):
        R = (R @ A).astype(bool)
    return R


class ContactDetector:
    """Finds self-contact clusters; caches the k-ring table per topology."""

    def __init__(self, k_ring=4, normal_dot=-0.5):
        self.k = k_ring
        self.ndot = normal_dot
        self._faces_id = None
        self._R = None

    def find(self, mem, d_c):
        """Return contact clusters as a list of vertex-index arrays,
        largest first (empty list: no contact)."""
        X, F = mem.X, mem.faces
        if id(F) != self._faces_id:            # topology changed -> rebuild
            self._R = _khop(F, len(X), self.k)
            self._faces_id = id(F)
        pairs = cKDTree(X).query_pairs(d_c, output_type="ndarray")
        if len(pairs) == 0:
            return []
        near = np.asarray(self._R[pairs[:, 0], pairs[:, 1]]).ravel()
        pairs = pairs[~near.astype(bool)]
        if len(pairs) == 0:
            return []
        vn = vertex_normals(X, F)
        dots = np.einsum('ij,ij->i', vn[pairs[:, 0]], vn[pairs[:, 1]])
        pairs = pairs[dots < self.ndot]
        if len(pairs) == 0:
            return []
        # cluster over contact pairs + mesh edges among flagged vertices
        flagged = np.unique(pairs)
        fmask = np.zeros(len(X), bool)
        fmask[flagged] = True
        me = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        me = me[fmask[me].all(axis=1)]
        gi = np.concatenate([pairs[:, 0], me[:, 0]])
        gj = np.concatenate([pairs[:, 1], me[:, 1]])
        sub = np.full(len(X), -1)
        sub[flagged] = np.arange(len(flagged))
        G = sparse.csr_matrix(
            (np.ones(len(gi), bool), (sub[gi], sub[gj])),
            shape=(len(flagged), len(flagged)))
        ncomp, lab = sparse.csgraph.connected_components(G, directed=False)
        clusters = [flagged[lab == c] for c in range(ncomp)]
        return sorted(clusters, key=len, reverse=True)


def _selftest_detector():
    """Two parallel sheets must flag; a lone sphere must not."""
    from membrane import Membrane, icosphere
    V, F = icosphere(3)
    mem = Membrane(V, F)
    det = ContactDetector()
    assert det.find(mem, 0.05) == []           # sphere: no self-contact
    # squash the sphere to an extreme pancake: top/bottom sheets nearly touch
    Xs = V.copy()
    Xs[:, 2] *= 0.01
    mem2 = Membrane(Xs, F)
    cl = ContactDetector().find(mem2, 0.05)
    assert len(cl) >= 1 and len(cl[0]) > 10
    # the flagged cluster must span both sheets (z of both signs)
    z = Xs[cl[0], 2]
    assert (z > 1e-6).any() and (z < -1e-6).any()


_selftest_detector()
