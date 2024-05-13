import gudhi as gd
import numpy as np

import multipers as mp

mp.simplex_tree_multi.SAFE_CONVERSION = True


def test_h1_rips_density():
    num_pts = 100
    dim = 2
    pts = np.random.uniform(size=(num_pts, dim))
    weights = np.random.uniform(size=num_pts)
    st = gd.RipsComplex(points=pts).create_simplex_tree()
    st = mp.SimplexTreeMulti(st, num_parameters=2)
    st.fill_lowerstar(weights, parameter=1)
    st.collapse_edges(num=100)
    st.expansion(2)
    # F=st.get_filtration_grid()
    # st.grid_squeeze(F, coordinate_values=True)
    (sm,) = mp.signed_measure(st, degree=1, plot=True, mass_default=None)
    pts, weights = sm
    sm_indices, unmappable_points = st.pts_to_indices(pts, simplices_dimensions=[1, 0])
    assert len(unmappable_points) == 0, "Found unmappable points in Rips edges ?"
    filtration_values = np.array([f for _, f in st.get_simplices()])
    reconstructed_measure = np.asarray(
        [
            [filtration_values[i][parameter] for parameter, i in enumerate(indices)]
            for indices in sm_indices
        ]
    )
    assert np.array_equal(
        reconstructed_measure, pts
    ), "Reconstructed measure is not equal to original measure ?"


def test_h0_rips_density():
    num_pts = 100
    dim = 2
    pts = np.random.uniform(size=(num_pts, dim))
    weights = np.random.uniform(size=num_pts)
    st = gd.RipsComplex(points=pts).create_simplex_tree()
    st = mp.SimplexTreeMulti(st, num_parameters=2)
    st.fill_lowerstar(weights, parameter=1)
    st.collapse_edges(full=True)
    # F=st.get_filtration_grid()
    # st.grid_squeeze(F, coordinate_values=True)
    (sm,) = mp.signed_measure(st, degree=0, plot=True, mass_default=None)
    pts, weights = sm
    _, unmappable_points = st.pts_to_indices(pts, simplices_dimensions=[1, 0])
    assert (
        pts[unmappable_points[:, 0]][:, 0] == 0
    ).all(), "Unmapped points of H0 have to be the nodes of the rips."


# def test_h1_rips_density_rank():
# 	num_pts = 100
# 	dim=2
# 	pts = np.random.uniform(size=(num_pts,dim))
# 	weights = np.random.uniform(size=num_pts)
# 	st = gd.RipsComplex(points=pts).create_simplex_tree()
# 	st = mp.SimplexTreeMulti(st, num_parameters=2)
# 	st.fill_lowerstar(weights, parameter=1)
# 	st.collapse_edges(full=True)
# 	st.expansion(2)
# 	# F=st.get_filtration_grid()
# 	# st.grid_squeeze(F, coordinate_values=True)
# 	sm, = mp.signed_measure(st, degree=1, plot=True, mass_default=None, invariant="rank_invariant", resolution=20, grid_strategy="quantile")
# 	pts, weights = sm
# 	sm_indices, unmappable_points = signed_measure_indices(st,pts, simplices_dimensions = [1,0])
# 	assert len(unmappable_points) == 0,  'Found unmappable points in Rips edges ?'
# 	filtration_values = np.array([f for _,f in st.get_simplices()])
# 	reconstructed_measure = np.asarray([[filtration_values[i][parameter%2] for parameter,i in enumerate(indices)] for indices in sm_indices])
# 	assert np.array_equal(reconstructed_measure, pts), 'Reconstructed measure is not equal to original measure ?'
