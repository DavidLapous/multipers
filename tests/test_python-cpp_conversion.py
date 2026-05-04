import gudhi as gd
import numpy as np

import multipers as mp


def test_random_alpha_conversion():
    x = np.random.uniform(size=(200, 2))
    num_parameter = 4
    st_gudhi = gd.AlphaComplex(points=x).create_simplex_tree()
    st_multi = mp.SimplexTreeMulti(st_gudhi, num_parameters=num_parameter)
    assert (
        np.all([s in st_multi for s, f in st_gudhi.get_simplices()])
        and st_gudhi.num_simplices() == st_multi.num_simplices
    ), "Simplices conversion failed."
    assert np.all([f.shape[0] == num_parameter for _, f in st_multi.get_simplices()]), (
        "Number of parameters is inconcistent"
    )
    assert np.all(
        [np.isclose(st_multi.filtration(s)[0], f) for s, f in st_gudhi.get_simplices()]
    ), "Filtration values conversion failed."


def test_random_alpha_conversion_default_values():
    x = np.random.uniform(size=(200, 2))
    num_parameter = 4
    st_gudhi = gd.AlphaComplex(points=x).create_simplex_tree()
    st_multi = mp.SimplexTreeMulti(
        st_gudhi, num_parameters=num_parameter, default_values=[1, 2, 3]
    )
    assert (
        np.all([s in st_multi for s, f in st_gudhi.get_simplices()])
        and st_gudhi.num_simplices() == st_multi.num_simplices
    ), "Simplices conversion failed."
    assert np.all([f.shape[0] == num_parameter for _, f in st_multi.get_simplices()]), (
        "Number of parameters is inconcistent"
    )
    assert np.all(
        [
            np.isclose(st_multi.filtration(s), [f, 1, 2, 3]).all()
            for s, f in st_gudhi.get_simplices()
        ]
    ), "Filtration values conversion failed."


def test_random_rips_conversion():
    x = np.random.uniform(size=(100, 2))
    num_parameter = 4
    st_gudhi = gd.RipsComplex(points=x).create_simplex_tree()
    st_multi = mp.SimplexTreeMulti(st_gudhi, num_parameters=num_parameter)
    assert (
        np.all([s in st_multi for s, f in st_gudhi.get_simplices()])
        and st_gudhi.num_simplices() == st_multi.num_simplices
    ), "Simplices conversion failed."
    assert np.all([f.shape[0] == num_parameter for _, f in st_multi.get_simplices()]), (
        "Number of parameters is inconcistent"
    )
    assert np.all(
        [np.isclose(st_multi.filtration(s)[0], f) for s, f in st_gudhi.get_simplices()]
    ), "Filtration values conversion failed."


def test_random_rips_conversion_default_values():
    x = np.random.uniform(size=(100, 2))
    num_parameter = 4
    st_gudhi = gd.RipsComplex(points=x).create_simplex_tree()
    st_multi = mp.SimplexTreeMulti(
        st_gudhi, num_parameters=num_parameter, default_values=[1, 2, 3]
    )
    assert (
        np.all([s in st_multi for s, f in st_gudhi.get_simplices()])
        and st_gudhi.num_simplices() == st_multi.num_simplices
    ), "Simplices conversion failed."
    assert np.all([f.shape[0] == num_parameter for _, f in st_multi.get_simplices()]), (
        "Number of parameters is inconcistent"
    )
    assert np.all(
        [
            np.isclose(st_multi.filtration(s), [f, 1, 2, 3]).all()
            for s, f in st_gudhi.get_simplices()
        ]
    ), "Filtration values conversion failed."
