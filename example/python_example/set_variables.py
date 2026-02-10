####################################################################################
# Copyright 2013 Daniel Albach, Erik Zenker, Carlchristian Eckert
#
# This file is part of HASEonGPU
#
# HASEonGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HASEonGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HASEonGPU.
# If not, see <http://www.gnu.org/licenses/>.
###################################################################################
###################################################################################
# set_variable.py (HASEonGPU)
# - Compacts/remaps points so all points are referenced by triangles
# - Writes parser-safe arrays:
#     triangleNormalPoint: unsigned int in [0 .. numberOfPoints-1] with max == numberOfPoints-1
#     triangleNeighbors:   int in [-1 .. numberOfTriangles-1] with exact min/max
#     forbiddenEdge:       int in [-1 .. 2] with exact min/max
####################################################################################

import numpy as np
import scipy.io
from scipy.io import savemat


def compact_mesh(p, t):
    """
    p: (N,2) float
    t: (M,3) int, 1-based (MATLAB style)
    Returns:
      p2: compacted points (K,2)
      t2: remapped triangles (M,3) still 1-based
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)

    # Flatten triangle indices, unique used vertices (still 1-based)
    used = np.unique(t.reshape(-1))
    used_sorted = np.sort(used)

    # Build mapping old_index(1-based) -> new_index(1-based)
    # Use dict for clarity; could be vectorized but this is robust.
    mapping = {old: new for new, old in enumerate(used_sorted, start=1)}

    # Remap triangles
    t2 = np.vectorize(mapping.get)(t).astype(np.int32)

    # Subset points (convert used indices to 0-based for slicing)
    p2 = p[used_sorted - 1, :].astype(np.float64)

    return p2, t2


def set_variables(p, t):
    """
    Builds variable.mat in a way that matches parser.cu assertions.
    """
    # 1) Compact mesh so numberOfPoints matches max index used by triangles
    p, t = compact_mesh(p, t)

    # Sanitize
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.int32)

    num_tri = t.shape[0]
    num_pts = p.shape[0]

    # Convert triangles to 0-based for computations
    t0 = t - 1  # shape (num_tri, 3), values 0..num_pts-1

    # =========================================================
    # STEP 1: triangleNeighbors (sorted_int)
    # MATLAB: neighbors stored 1..num_tri, 0 means "none"
    # We will finally store as int32 with -1 meaning none, else 0..num_tri-1
    # =========================================================
    sorted_int = np.zeros((num_tri, 3), dtype=np.int32)  # 0 means "no neighbor" (MATLAB style)

    for i in range(num_tri):
        a, b, c = t0[i]
        for j in range(num_tri):
            if i == j:
                continue
            tj = t0[j]
            # Share edge (a,b)
            if len({a, b}.intersection(tj)) == 2:
                sorted_int[i, 0] = j + 1
            # Share edge (a,c)
            if len({a, c}.intersection(tj)) == 2:
                sorted_int[i, 1] = j + 1
            # Share edge (b,c)
            if len({b, c}.intersection(tj)) == 2:
                sorted_int[i, 2] = j + 1

    # Convert neighbors to int32 with -1 for boundary, else 0..num_tri-1
    triangleNeighbors = (sorted_int - 1).astype(np.int32)  # 0->-1, (j+1)->j

    # =========================================================
    # STEP 2: forbiddenEdge
    # MATLAB: forbidden in {0,1,2,3} where 0 = none
    # parser.cu wants int range exactly [-1..2] (equals==true)
    # We'll store -1 for none, else 0..2
    # =========================================================
    forbidden = np.zeros((num_tri, 3), dtype=np.int32)  # MATLAB style: 0 means none, else 1..3

    for i in range(num_tri):
        for k in range(3):
            face = sorted_int[i, k] - 1  # neighbor triangle index (0-based) or -1
            if face < 0:
                continue
            # find which edge index (1..3) in neighbor points back to i
            for kk in range(3):
                if (sorted_int[face, kk] - 1) == i:
                    forbidden[i, k] = kk + 1
                    break

    forbiddenEdge = (forbidden - 1).astype(np.int32)  # 0->-1, 1..3 -> 0..2

    # =========================================================
    # STEP 3: triangle centers
    # =========================================================
    x_center = np.zeros(num_tri, dtype=np.float64)
    y_center = np.zeros(num_tri, dtype=np.float64)
    for i in range(num_tri):
        pts = t0[i]
        x_center[i] = np.mean(p[pts, 0])
        y_center[i] = np.mean(p[pts, 1])

    # =========================================================
    # STEP 4: normals + triangleNormalPoint
    # parser.cu expects triangleNormalPoint unsigned with min=0 max=num_pts-1 (equals true)
    #
    # We'll follow the original logic:
    #  edge 1-2 uses point 1
    #  edge 1-3 uses point 1
    #  edge 2-3 uses point 2
    #
    # IMPORTANT: after compaction, max point index WILL be used somewhere (because points are only used ones)
    # =========================================================
    normals_x = np.zeros((num_tri, 3), dtype=np.float64)
    normals_y = np.zeros((num_tri, 3), dtype=np.float64)
    normals_z = np.zeros((num_tri, 3), dtype=np.float64)

    triangleNormalPoint = np.zeros((num_tri, 3), dtype=np.uint32)

    vec2 = np.array([0.0, 0.0, 0.1], dtype=np.float64)

    for i in range(num_tri):
        pts = t0[i]

        # Edge 1-2
        vec1 = np.array([p[pts[0], 0], p[pts[0], 1], 0.0], dtype=np.float64) - \
               np.array([p[pts[1], 0], p[pts[1], 1], 0.0], dtype=np.float64)
        n = np.cross(vec1, vec2)
        n /= np.linalg.norm(n)
        normals_x[i, 0], normals_y[i, 0], normals_z[i, 0] = n
        triangleNormalPoint[i, 0] = np.uint32(pts[0])

        # Edge 1-3
        vec1 = np.array([p[pts[0], 0], p[pts[0], 1], 0.0], dtype=np.float64) - \
               np.array([p[pts[2], 0], p[pts[2], 1], 0.0], dtype=np.float64)
        n = np.cross(vec1, vec2)
        n /= np.linalg.norm(n)
        normals_x[i, 1], normals_y[i, 1], normals_z[i, 1] = n
        triangleNormalPoint[i, 1] = np.uint32(pts[0])

        # Edge 2-3
        vec1 = np.array([p[pts[1], 0], p[pts[1], 1], 0.0], dtype=np.float64) - \
               np.array([p[pts[2], 0], p[pts[2], 1], 0.0], dtype=np.float64)
        n = np.cross(vec1, vec2)
        n /= np.linalg.norm(n)
        normals_x[i, 2], normals_y[i, 2], normals_z[i, 2] = n
        triangleNormalPoint[i, 2] = np.uint32(pts[1])

    # =========================================================
    # STEP 5: triangle surface (MATLAB formula)
    # =========================================================
    surface = np.zeros(num_tri, dtype=np.float64)
    for i in range(num_tri):
        a = np.sum((p[t0[i, 0]] - p[t0[i, 1]]) ** 2)
        b = np.sum((p[t0[i, 2]] - p[t0[i, 1]]) ** 2)
        c = np.sum((p[t0[i, 0]] - p[t0[i, 2]]) ** 2)
        surface[i] = np.sqrt(1.0 / 16.0 * (4.0 * a * c - (a + c - b) ** 2))

    # =========================================================
    # Save MAT
    # Notes:
    # - Keep triangleNeighbors / forbiddenEdge SIGNED int32 because parser expects -1 allowed.
    # - triangleNormalPoint must be uint32 0..num_pts-1 with exact max == num_pts-1.
    # - trianglePointIndices ("t_int") should be uint32 0..num_pts-1.
    # =========================================================
    savemat("variable.mat", {
        "t_int": t0.astype(np.uint32),
        "sorted_int": triangleNeighbors.astype(np.int32),
        "forbidden": forbiddenEdge.astype(np.int32),
        "normals_p": triangleNormalPoint.astype(np.uint32),

        "normals_x": normals_x,
        "normals_y": normals_y,
        "normals_z": normals_z,
        "x_center": x_center,
        "y_center": y_center,
        "surface": surface,

        # Optional: save compacted p/t too for debugging
        "p_compact": p,
        "t_compact": t
    })

    # Quick diagnostics (helps confirm parser assertions)
    tp = triangleNormalPoint.reshape(-1)
    tn = triangleNeighbors.reshape(-1)
    fe = forbiddenEdge.reshape(-1)

    print("variable.mat generated (parser-safe).")
    print(f"[check] numberOfPoints={num_pts}  numberOfTriangles={num_tri}")
    print(f"[check] triangleNormalPoint min={tp.min()} max={tp.max()} (expect 0..{num_pts-1})")
    print(f"[check] triangleNeighbors   min={tn.min()} max={tn.max()} (expect -1..{num_tri-1})")
    print(f"[check] forbiddenEdge       min={fe.min()} max={fe.max()} (expect -1..2)")


if __name__ == "__main__":
    mat = scipy.io.loadmat("pt.mat")
    set_variables(mat["p"], mat["t"])
