import numpy as np

import multipers as mp
from multipers import signed_measure

# def test_1(): # TODO: test integrate_measure instead
#     st = mp.SimplexTreeMulti(num_parameters=3)
#     st.insert([0], [1, 0, 0])
#     st.insert([1], [0, 1, 0])
#     st.insert([2], [0, 0, 1])
#     st.insert([0, 1, 2], [2, 2, 2])
#     st.make_filtration_non_decreasing()
#     st = st.grid_squeeze(grid_strategy="exact")
#     assert np.array_equal(
#         hilbert_surface(st, degrees=[0])[1],
#         np.array(
#             [
#                 [
#                     [[0, 1, 1], [1, 2, 2], [1, 2, 2]],
#                     [[1, 2, 2], [2, 3, 3], [2, 3, 3]],
#                     [[1, 2, 2], [2, 3, 3], [2, 3, 1]],
#                 ]
#             ]
#         ),
#     )
#     assert np.array_equal(hilbert_surface(st, degrees=[0])[1][0], euler_surface(st)[1])


# def test_2():
#     st = mp.SimplexTreeMulti(num_parameters=4)
#     st.insert([0], [1, 0, 0, 0])
#     st.insert([1], [0, 1, 0, 0])
#     st.insert([2], [0, 0, 1, 0])
#     st.insert([3], [0, 0, 0, 1])
#     st.insert([0, 1, 2, 3], [2, 2, 2, 2])
#     st.make_filtration_non_decreasing()
#     # list(st.get_simplices())
#     st.grid_squeeze(grid_strategy="exact")
#     assert np.array_equal(
#         hilbert_surface(st, degrees=[0])[1][0], (euler_surface(st)[1])
#     )


def test_3():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0, 1, 2], [1] * st.num_parameters)
    st.remove_maximal_simplex([0, 1, 2])
    st = st.grid_squeeze(strategy="exact")
    ((a, b),) = mp.signed_measure(st, degrees=[1], mass_default=None)
    assert np.array_equal(a, [[1, 1]]) and np.array_equal(b, [1])
    assert mp.signed_measure(st, degrees=[1], mass_default="inf")[0][1].sum() == 0


def test_4():
    st = mp.SimplexTreeMulti(num_parameters=3)
    st.insert([0], [1, 0, 0])
    st.insert([1], [0, 1, 0])
    st.insert([2], [0, 0, 1])
    st.insert([0, 1, 2], [2, 2, 2])
    st.make_filtration_non_decreasing()
    # list(st.get_simplices())
    st = st.grid_squeeze(strategy="exact")
    assert signed_measure(st, degrees=[0], mass_default="inf")[0][1].sum() == 0


def test_5():
    num_param = 7
    st = mp.SimplexTreeMulti(num_parameters=num_param)
    for i in range(num_param):
        f = np.zeros(num_param)
        f[i] = 1
        st.insert([i], f)
    st.insert(np.arange(num_param), [2] * num_param)
    assert not st.make_filtration_non_decreasing()
    st = st.grid_squeeze()
    (a, b), (c, d) = signed_measure(st, degrees=[0, 1])
    assert np.all(a[-1] == 2)
    assert np.sum(b) == 1 and b[-1] == -(num_param - 1)
    assert c.shape == (0, num_param)
