# =====================================================================
# PERSON 1 - CELL REPLACEMENTS (notebook cells 12, 13, 14, 15, 18)
# Paste each block into the notebook cell with the same number.
# Cells 10, 11 and 17 are UNCHANGED. No new cells are inserted, so the
# cell numbers used by Persons 2-5 stay valid.
# =====================================================================

# %% [Cell 12 - code] (replaces the print cell; keeps the original prints, adds integrity checks)
print("Subject IDs stored in all_data:", all_data.keys())
random_sub_id = random.choice(list(all_data.keys()))
print(f"Vertebrae available for subject {random_sub_id}:", all_data[random_sub_id].keys())
print(f"Data fields stored for subject {random_sub_id}, vertebra L1:", all_data[random_sub_id]['L1'].keys())

# ---- Data integrity checks on the RAW loaded meshes ----
assert len(all_data) == n_subjects
reference_faces = all_data[1]['L1']['face']          # connectivity of the first mesh
for sid, sub in all_data.items():
    assert list(sub.keys()) == vertebrae_classes, f"unexpected classes for subject {sid}"
    for cls, d in sub.items():
        assert d['points'].shape == (4000, 3), f"{cls}_{sid:02d}: wrong point count"
        assert np.isfinite(d['points']).all(), f"{cls}_{sid:02d}: NaN/inf in points"
        # identical triangle lists => the 4,000 points correspond across all meshes
        assert np.array_equal(d['face'], reference_faces), f"{cls}_{sid:02d}: connectivity differs"
print(f"Integrity OK: {len(all_data) * len(vertebrae_classes)} meshes, 4000 points each, "
      f"{reference_faces.shape[0]} identical triangles, no NaN/inf.")

# %% [Cell 13 - MARKDOWN] replace the last sentence of the cell with:
# """This function translates the 3D point cloud to the origin and scales it uniformly so that the
# farthest point lies at distance 1 from the origin, i.e. every shape fits inside the unit sphere."""

# %% [Cell 14 - code] (function body unchanged; docstring corrected; unit tests added)
## Normalising Function

def center_and_l2_normalize(points):
    """
    Translate the point-cloud centroid to (0,0,0) and scale the whole shape
    uniformly by the maximum Euclidean norm of the centred points, so that
    the farthest point lies at distance 1 (unit sphere).

    Args:
        points (numpy.ndarray): shape (N, 3) point cloud

    Returns:
        numpy.ndarray: transformed point cloud, shape (N, 3)
    """
    # Step 1: move centroid to origin
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Step 2: find max distance from origin
    distances = np.linalg.norm(centered, axis=1)
    max_dist = np.max(distances)

    # Step 3: scale so the farthest point lies on the unit sphere
    normalized = centered / (max_dist + 1e-8)  # avoid divide by zero

    return normalized


# ---- Unit tests ----
_rng = np.random.default_rng(0)
_P = _rng.normal(size=(500, 3)) * [3.0, 1.0, 0.5] + [10.0, -4.0, 7.0]
_out = center_and_l2_normalize(_P)
assert np.allclose(_out.mean(axis=0), 0, atol=1e-6), "centroid is not at the origin"
assert np.isclose(np.linalg.norm(_out, axis=1).max(), 1.0, atol=1e-6), "max radius is not 1"
assert np.allclose(center_and_l2_normalize(_P + [5.0, 5.0, -9.0]), _out, atol=1e-6), "not translation invariant"
assert np.allclose(center_and_l2_normalize(_P * 37.0), _out, atol=1e-6), "not scale invariant"
assert np.allclose(center_and_l2_normalize(np.ones((10, 3))), 0), "degenerate cloud should map to zeros"
print("center_and_l2_normalize: all 5 unit tests passed.")

# %% [Cell 15 - code] (fixes the shallow-copy bug: all_data now keeps the RAW points)
## Normalising all the points of the dataset using center_and_l2_normalize function

all_data_norm = {
    sub_id: {
        vert_class: {
            'points': center_and_l2_normalize(sub[vert_class]['points']),  # new array
            'face':   sub[vert_class]['face'],                             # connectivity (read-only)
        }
        for vert_class in vertebrae_classes
    }
    for sub_id, sub in all_data.items()
}

# raw data must be untouched, normalised data must differ from it
assert not np.allclose(all_data[1]['L1']['points'], all_data_norm[1]['L1']['points'])

# %% [Cell 18 - code] (keeps the original prints; adds checks that Persons 2-5 rely on)
# Added explanations
print("Shape of X_array (number of samples × flattened vertex coordinates):", X_array.shape)
print("Shape of Y_array (number of labels):", Y_array.shape)

# ---- Integrity checks ----
assert X_array.shape == (150, 12000) and Y_array.shape == (150,)
assert np.isfinite(X_array).all()
assert np.array_equal(np.bincount(Y_array), [30] * 5), "classes are not balanced (30 per class)"

# Row-order contract (needed for subject_ids = np.repeat(keys, 5) in Cell 27):
# rows are subject-major; within each subject the labels are 0,1,2,3,4 (L1..L5).
assert np.array_equal(Y_array, np.tile(np.arange(5), n_subjects)), "row order is not subject-major / L1..L5"
_subject_ids = np.repeat(np.asarray(list(all_data_norm.keys())), len(vertebrae_classes))
for _sid in np.unique(_subject_ids):
    assert sorted(Y_array[_subject_ids == _sid]) == [0, 1, 2, 3, 4]

# Every row must be exactly the flattened (4000, 3) normalised mesh it claims to be
_row = 0
for _sid in all_data_norm.keys():
    for _cls in vertebrae_classes:
        assert np.array_equal(X_array[_row], all_data_norm[_sid][_cls]['points'].flatten())
        _row += 1

# Every shape is centred at the origin and has max radius 1
_P = X_array.reshape(150, 4000, 3)
# tolerance 1e-4: pv.read returns float32, whose rounding leaves a centroid residual of ~1e-5
# (about 0.001% of the shape size, negligible next to the ~0.04 RMS shape variation)
assert np.abs(_P.mean(axis=1)).max() < 1e-4
assert np.allclose(np.linalg.norm(_P, axis=2).max(axis=1), 1.0, atol=1e-5)

print("Class counts:", {c: int(n) for c, n in zip(vertebrae_classes, np.bincount(Y_array))})
print("All integrity checks passed: X_array (150, 12000), Y_array (150,), 30 shapes per class, subject-major order.")
