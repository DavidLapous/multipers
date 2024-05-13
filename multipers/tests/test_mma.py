import numpy as np

import multipers as mp

mp.simplex_tree_multi.SAFE_CONVERSION = True


def test_1():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    mma_pymodule = st.persistence_approximation()
    assert np.array_equal(mma_pymodule[0].get_birth_list(), [[1.0, 0.0], [0.0, 1.0]])
    assert np.array_equal(mma_pymodule[0].get_death_list(), [[np.inf, np.inf]])


def test_img():
    simplextree = mp.SimplexTreeMulti(num_parameters=4)
    simplextree.insert([0], [1, 2, 3, 4])
    mod = simplextree.persistence_approximation(
        box=[[0, 0, 0, 0], [5, 5, 5, 5]], max_error=1.0
    )
    img = mod.representation(resolution=6, kernel="linear")
    assert np.isclose(img[0, 2, 3, 4, 5], 0.5)
    assert np.isclose(img[0, 1, 1, 1, 1], 0)
    assert np.isclose(img[0, 3, 4, 5, 5], 1)


# def test_2():
#     st = mp.SimplexTreeMulti(num_parameters=2)
#     st.insert([0], [1, 2])
#     st.insert([0, 1], [2, 2])
#     mod = st.persistence_approximation(
#         box=[[0, 0], [4, 4]], slicer_backend="graph", max_error=0.1
#     )
#     assert len(mod) == 1
#     assert np.isclose(
#         mod.representation(degrees=[0], plot=False, resolution=5, kernel="linear", p=1),
#         np.array(
#             [
#                 [
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 1, 1, 1, 1],
#                     [0, 1, 2, 2, 2],
#                     [0, 1, 2, 2, 2],
#                 ]
#             ]
#         ),
#     ).all()
