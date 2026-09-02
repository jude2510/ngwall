"""Dependency-free VTK XML output: ParaView time series for wall meshes.

Writes one PolyData (.vtp) file per snapshot -- multiple walls merged into a
single piece with offset connectivity, so walls appear/disappear correctly
under one ParaView source -- plus a .pvd collection index mapping each file
to its simulation time (ParaView's timeline then shows real t).

ASCII format by choice: human-inspectable, no encoding pitfalls, and at our
mesh sizes (~2.5k vertices) a frame is ~250 KB.

Usage:
    series = VtkSeries("kick_vtk", "kick")
    series.add(t, [wall_block(mem.X, mem.faces, speed, wall_id=0), ...])
    ...
    print(series.close())    # path of the .pvd to open in ParaView
"""

import os
import numpy as np


def wall_block(X, faces, speed, wall_id):
    """Bundle one wall's mesh + per-vertex arrays (speed, gamma, wall id)."""
    gam = 1.0 / np.sqrt(np.clip(1.0 - speed ** 2, 1e-14, None))
    pd = {"speed": np.asarray(speed, np.float32),
          "gamma": gam.astype(np.float32),
          "wall": np.full(len(X), wall_id, np.int32)}
    return (np.asarray(X, np.float32), np.asarray(faces), pd)


def _ascii(a):
    # Flush float32 subnormals (|x| < FLT_MIN) to 0: VTK's ASCII reader parses
    # with C++ istream >> float, which sets failbit on subnormal underflow
    # (libc++) and then reports the array as "too short". Such values (~1e-40
    # at a collapsing axis) are numerical zero for our O(1) geometry.
    return " ".join("0" if 0.0 < abs(x) < 1.2e-38 else f"{x:.7g}"
                    for x in np.asarray(a).ravel())


def _vtp_xml(blocks):
    """XML string of one PolyData with all blocks merged."""
    names = list(blocks[0][2])
    X = np.concatenate([b[0] for b in blocks])
    off = np.cumsum([0] + [len(b[0]) for b in blocks])
    F = np.concatenate([b[1] + off[i] for i, b in enumerate(blocks)])
    pd = {k: np.concatenate([b[2][k] for b in blocks]) for k in names}
    npt, npoly = len(X), len(F)
    parts = [
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">',
        '<PolyData>',
        f'<Piece NumberOfPoints="{npt}" NumberOfVerts="0" NumberOfLines="0" '
        f'NumberOfStrips="0" NumberOfPolys="{npoly}">',
        '<Points>',
        '<DataArray type="Float32" NumberOfComponents="3" format="ascii">',
        _ascii(X), '</DataArray>', '</Points>',
        f'<PointData Scalars="{names[0]}">']
    for k in names:
        typ = "Int32" if pd[k].dtype.kind == "i" else "Float32"
        parts += [f'<DataArray type="{typ}" Name="{k}" format="ascii">',
                  _ascii(pd[k]), '</DataArray>']
    parts += [
        '</PointData>', '<Polys>',
        '<DataArray type="Int32" Name="connectivity" format="ascii">',
        _ascii(F), '</DataArray>',
        '<DataArray type="Int32" Name="offsets" format="ascii">',
        _ascii(np.arange(3, 3 * npoly + 1, 3)), '</DataArray>',
        '</Polys>', '</Piece>', '</PolyData>', '</VTKFile>']
    return "\n".join(parts)


class VtkSeries:
    def __init__(self, outdir, name):
        os.makedirs(outdir, exist_ok=True)
        self.dir, self.name = outdir, name
        self.frames = []          # (t, filename)

    def add(self, t, blocks):
        fn = f"{self.name}_{len(self.frames):04d}.vtp"
        with open(os.path.join(self.dir, fn), "w") as f:
            f.write(_vtp_xml(blocks))
        self.frames.append((t, fn))
        self._write_pvd()        # keep the index valid mid-run

    def last_t(self):
        return self.frames[-1][0] if self.frames else -np.inf

    def _write_pvd(self):
        path = os.path.join(self.dir, f"{self.name}.pvd")
        rows = "\n".join(
            f'<DataSet timestep="{t:.6f}" group="" part="0" file="{fn}"/>'
            for t, fn in self.frames)
        with open(path, "w") as f:
            f.write('<VTKFile type="Collection" version="0.1" '
                    'byte_order="LittleEndian">\n<Collection>\n'
                    f'{rows}\n</Collection>\n</VTKFile>\n')
        return path

    def close(self):
        return self._write_pvd()


def _selftest_xml():
    """Two-block merge must be well-formed XML with consistent counts."""
    import xml.etree.ElementTree as ET
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2]])
    blocks = [wall_block(X, F, np.zeros(3), 0),
              wall_block(X + 2, F, np.full(3, 0.99), 1)]
    root = ET.fromstring(_vtp_xml(blocks))
    piece = root.find("PolyData/Piece")
    assert piece.get("NumberOfPoints") == "6"
    assert piece.get("NumberOfPolys") == "2"
    conn = piece.find("Polys/DataArray").text.split()
    assert conn == ["0", "1", "2", "3", "4", "5"]
    names = [a.get("Name") for a in piece.find("PointData")]
    assert names == ["speed", "gamma", "wall"]


_selftest_xml()
