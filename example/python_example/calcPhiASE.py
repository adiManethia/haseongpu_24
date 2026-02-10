import numpy as np
import os
import warnings
import shutil

############################## clean_IO_files #################################
def clean_IO_files(TMP_FOLDER):
    if os.path.exists(TMP_FOLDER) and os.path.isdir(TMP_FOLDER):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shutil.rmtree(TMP_FOLDER)


########################### mesh compaction for parser ########################
def _compact_mesh_for_parser(points,
                             trianglePointIndices,
                             triangleNormalPoint,
                             betaCells):
    """
    Compacts/remaps points so that:
      - numberOfPoints equals the number of used vertices in trianglePointIndices
      - triangleNormalPoint max == numberOfPoints-1 (as required by parser.cu)
    Also compacts betaCells accordingly (per-point x levels).

    Expected shapes coming from laserPumpCladdingExample.py:
      points: (N,2)
      trianglePointIndices: (T,3) or (3,T) (0-based indices)
      triangleNormalPoint: (T,3) or (3,T) (0-based indices)
      betaCells: (N,levels)

    Returns compacted versions in the SAME orientation as inputs were provided.
    """

    # Keep track of whether triangles were provided transposed
    tri_was_transposed = False
    tnp_was_transposed = False

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N,2), got {pts.shape}")

    tri = np.asarray(trianglePointIndices)
    if tri.ndim != 2:
        raise ValueError(f"trianglePointIndices must be 2D, got {tri.shape}")
    if tri.shape[0] == 3 and tri.shape[1] != 3:
        # likely (3,T)
        tri = tri.T
        tri_was_transposed = True
    elif tri.shape[1] != 3:
        raise ValueError(f"trianglePointIndices must be (T,3) or (3,T), got {trianglePointIndices.shape}")

    tnp = np.asarray(triangleNormalPoint)
    if tnp.ndim != 2:
        raise ValueError(f"triangleNormalPoint must be 2D, got {tnp.shape}")
    if tnp.shape[0] == 3 and tnp.shape[1] != 3:
        tnp = tnp.T
        tnp_was_transposed = True
    elif tnp.shape[1] != 3:
        raise ValueError(f"triangleNormalPoint must be (T,3) or (3,T), got {triangleNormalPoint.shape}")

    tri = tri.astype(np.int64, copy=False)
    tnp = tnp.astype(np.int64, copy=False)

    # Find used vertices from trianglePointIndices
    used = np.unique(tri.reshape(-1))
    used_sorted = np.sort(used)

    # Build old->new mapping (0-based)
    oldN = pts.shape[0]
    map_old_to_new = -np.ones(oldN, dtype=np.int64)
    map_old_to_new[used_sorted] = np.arange(used_sorted.size, dtype=np.int64)

    # Remap triangle indices and triangle normal point indices
    tri2 = map_old_to_new[tri]
    tnp2 = map_old_to_new[tnp]

    if (tri2 < 0).any():
        bad = np.unique(tri[tri2 < 0])
        raise ValueError(f"trianglePointIndices reference points outside points array. Bad indices: {bad[:10]}")
    if (tnp2 < 0).any():
        # This means triangleNormalPoint references points not used by triangles.
        # We can fix by snapping those to a valid used vertex (0), but better to fail loudly.
        bad = np.unique(tnp[tnp2 < 0])
        raise ValueError(
            "triangleNormalPoint references vertices not present in trianglePointIndices.\n"
            f"Bad indices (sample): {bad[:10]}\n"
            "Fix generation of triangleNormalPoint, or ensure it uses triangle vertices only."
        )

    # Compact points
    pts2 = pts[used_sorted, :]

    # Compact betaCells (per-point)
    bc = np.asarray(betaCells, dtype=np.float64)
    if bc.ndim != 2 or bc.shape[0] != oldN:
        raise ValueError(f"betaCells must be (N,levels) with N={oldN}, got {bc.shape}")
    bc2 = bc[used_sorted, :]

    # Restore original orientations if needed
    if tri_was_transposed:
        tri2 = tri2.T
    if tnp_was_transposed:
        tnp2 = tnp2.T

    return pts2, tri2.astype(np.uint32), tnp2.astype(np.uint32), bc2


########################### create_calcPhiASE_input ###########################
def create_calcPhiASE_input(points,
                            triangleNormalsX,
                            triangleNormalsY,
                            forbiddenEdge,
                            triangleNormalPoint,
                            triangleNeighbors,
                            trianglePointIndices,
                            thickness,
                            numberOfLevels,
                            nTot,
                            betaVolume,
                            laserParameter,
                            crystal,
                            betaCells,
                            triangleSurfaces,
                            triangleCenterX,
                            triangleCenterY,
                            claddingCellTypes,
                            claddingNumber,
                            claddingAbsorption,
                            refractiveIndices,
                            reflectivities,
                            FOLDER):

    CURRENT_DIR = os.getcwd()
    os.makedirs(FOLDER, exist_ok=True)
    os.chdir(FOLDER)

    # ensure arrays are transposed properly for saving (MATLAB/Fortran style flattening)
    points = np.transpose(points)  # (2,N)
    triangleNormalsX = np.transpose(triangleNormalsX)
    triangleNormalsY = np.transpose(triangleNormalsY)
    forbiddenEdge = np.transpose(forbiddenEdge)
    triangleNormalPoint = np.transpose(triangleNormalPoint)
    triangleNeighbors = np.transpose(triangleNeighbors)
    trianglePointIndices = np.transpose(trianglePointIndices)
    betaVolume = np.transpose(betaVolume)

    laserParameter['s_abs'] = np.transpose(laserParameter['s_abs'])
    laserParameter['s_ems'] = np.transpose(laserParameter['s_ems'])
    laserParameter['l_abs'] = np.transpose(laserParameter['l_abs'])
    laserParameter['l_ems'] = np.transpose(laserParameter['l_ems'])

    betaCells = np.transpose(betaCells)
    triangleSurfaces = np.transpose(triangleSurfaces)
    triangleCenterX = np.transpose(triangleCenterX)
    triangleCenterY = np.transpose(triangleCenterY)
    claddingCellTypes = np.transpose(claddingCellTypes)
    refractiveIndices = np.transpose(refractiveIndices)
    reflectivities = np.transpose(reflectivities)

    # save arrays as text files
    np.savetxt('points.txt', points, delimiter='\n', fmt='%.50f')
    np.savetxt('triangleNormalsX.txt', triangleNormalsX, delimiter='\n', fmt='%.50f')
    np.savetxt('triangleNormalsY.txt', triangleNormalsY, delimiter='\n', fmt='%.50f')

    # IMPORTANT TYPES:
    # forbiddenEdge and triangleNeighbors are allowed to contain -1 (parser expects that), so keep them signed.
    np.savetxt('forbiddenEdge.txt', forbiddenEdge, delimiter='\n', fmt='%d')
    np.savetxt('triangleNeighbors.txt', triangleNeighbors, delimiter='\n', fmt='%d')

    # These are unsigned indices, but saving as %d is fine as long as values are non-negative.
    np.savetxt('triangleNormalPoint.txt', triangleNormalPoint, delimiter='\n', fmt='%d')
    np.savetxt('trianglePointIndices.txt', trianglePointIndices, delimiter='\n', fmt='%d')

    with open('thickness.txt', 'w') as f:
        f.write(str(thickness) + '\n')
    with open('numberOfLevels.txt', 'w') as f:
        f.write(str(numberOfLevels) + '\n')
    with open('numberOfTriangles.txt', 'w') as f:
        f.write(str(trianglePointIndices.shape[1]) + '\n')
    with open('numberOfPoints.txt', 'w') as f:
        f.write(str(points.shape[1]) + '\n')

    with open('nTot.txt', 'w') as f:
        f.write(str(float(nTot)) + '\n')

    np.savetxt('betaVolume.txt', betaVolume, delimiter='\n', fmt='%.50f')
    np.savetxt('sigmaA.txt', laserParameter['s_abs'], delimiter='\n', fmt='%.50f')
    np.savetxt('sigmaE.txt', laserParameter['s_ems'], delimiter='\n', fmt='%.50f')
    np.savetxt('lambdaA.txt', laserParameter['l_abs'], delimiter='\n', fmt='%.50f')
    np.savetxt('lambdaE.txt', laserParameter['l_ems'], delimiter='\n', fmt='%.50f')

    with open('crystalTFluo.txt', 'w') as f:
        f.write(str(crystal['tfluo']) + '\n')

    np.savetxt('betaCells.txt', betaCells, delimiter='\n', fmt='%.50f')
    np.savetxt('triangleSurfaces.txt', triangleSurfaces, delimiter='\n', fmt='%.50f')
    np.savetxt('triangleCenterX.txt', triangleCenterX, delimiter='\n', fmt='%.50f')
    np.savetxt('triangleCenterY.txt', triangleCenterY, delimiter='\n', fmt='%.50f')

    np.savetxt('claddingCellTypes.txt', claddingCellTypes, delimiter='\n', fmt='%d')
    with open('claddingNumber.txt', 'w') as f:
        f.write(str(claddingNumber) + '\n')
    with open('claddingAbsorption.txt', 'w') as f:
        f.write(str(claddingAbsorption) + '\n')

    np.savetxt('refractiveIndices.txt', refractiveIndices, delimiter='\n', fmt='%3.5f')
    np.savetxt('reflectivities.txt', reflectivities, delimiter='\n', fmt='%.50f')

    os.chdir(CURRENT_DIR)


######################### parse_calcPhiASE_output #############################
def parse_calcPhiASE_output(FOLDER):
    import re
    CURRENT_DIR = os.getcwd()
    os.chdir(FOLDER)

    def _read_array(fname, dtype=float):
        with open(fname, "r") as fid:
            # First line: shape (may contain commas too)
            shape_tokens = next(fid).strip().replace(",", "").split()
            arraySize = [int(x) for x in shape_tokens]

            # Second line: values (may contain commas as thousands separators)
            line = next(fid).strip()
            # Remove commas inside numbers: "1,234" -> "1234"
            line = line.replace(",", "")
            # Split on whitespace
            vals = [dtype(x) for x in line.split()]
            arr = np.reshape(vals, arraySize, order="F")
            return arr

    phiASE = _read_array("phi_ASE.txt", float)
    mseValues = _read_array("mse_values.txt", float)
    raysUsedPerSample = _read_array("N_rays.txt", float)

    os.chdir(CURRENT_DIR)
    return [mseValues, raysUsedPerSample, phiASE]




################################ calcPhiASE ########################################
def calcPhiASE(
    points,
    trianglePointIndices,
    betaCells,
    betaVolume,
    claddingCellTypes,
    claddingNumber,
    claddingAbsorption,
    useReflections,
    refractiveIndices,
    reflectivities,
    triangleNormalsX,
    triangleNormalsY,
    triangleNeighbors,
    triangleSurfaces,
    triangleCenterX,
    triangleCenterY,
    triangleNormalPoint,
    forbiddenEdge,
    minRaysPerSample,
    maxRaysPerSample,
    mseThreshold,
    repetitions,
    nTot,
    thickness,
    laserParameter,
    crystal,
    numberOfLevels,
    deviceMode,
    parallelMode,
    maxGPUs,
    nPerNode
):

    # -------------------------
    # Ensure parser-safe mesh I/O by compacting/remapping points
    # -------------------------
    points2, tri2, tnp2, betaCells2 = _compact_mesh_for_parser(
        points=points,
        trianglePointIndices=trianglePointIndices,
        triangleNormalPoint=triangleNormalPoint,
        betaCells=betaCells
    )

    # Replace with compacted arrays for file writing and for maxSample calculation
    points = points2
    trianglePointIndices = tri2
    triangleNormalPoint = tnp2
    betaCells = betaCells2

    minSample = 0
    nP = points.shape[0]  # points is (N,2)
    maxSample = (numberOfLevels * nP) - 1

    REFLECT = ' --reflection=1' if useReflections else ' --reflection=0'

    Prefix = ''
    if parallelMode == 'mpi':
        Prefix = f"mpiexec -npernode {nPerNode} "
        maxGPUs = 1

    CALCPHIASE_DIR = os.getcwd()
    TMP_FOLDER = os.path.join(CALCPHIASE_DIR, 'input_tmp')

    clean_IO_files(TMP_FOLDER)

    create_calcPhiASE_input(
        points,
        triangleNormalsX,
        triangleNormalsY,
        forbiddenEdge,
        triangleNormalPoint,
        triangleNeighbors,
        trianglePointIndices,
        thickness,
        numberOfLevels,
        nTot,
        betaVolume,
        laserParameter,
        crystal,
        betaCells,
        triangleSurfaces,
        triangleCenterX,
        triangleCenterY,
        claddingCellTypes,
        claddingNumber,
        claddingAbsorption,
        refractiveIndices,
        reflectivities,
        TMP_FOLDER
    )

    status = os.system(
        Prefix + CALCPHIASE_DIR + '/../../build/calcPhiASE '
        + f'--parallel-mode={parallelMode}'
        + f' --device-mode={deviceMode}'
        + f' --min-rays={int(minRaysPerSample)}'
        + f' --max-rays={int(maxRaysPerSample)}'
        + REFLECT
        + f' --input-path={TMP_FOLDER}'
        + f' --output-path={TMP_FOLDER}'
        + f' --min-sample-i={minSample}'
        + f' --max-sample-i={maxSample}'
        + f' --ngpus={maxGPUs}'
        + f' --repetitions={repetitions}'
        + f' --mse-threshold={mseThreshold}'
        + f' --spectral-resolution={laserParameter["l_res"]}'
    )

    if status != 0:
        print('This step of the raytracing computation did NOT finish successfully. Aborting.')
        exit()

    mseValues, raysUsedPerSample, phiASE = parse_calcPhiASE_output(TMP_FOLDER)
    clean_IO_files(TMP_FOLDER)

    return phiASE, mseValues, raysUsedPerSample


  
