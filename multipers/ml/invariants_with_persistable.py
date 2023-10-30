import persistable


# requires installing ripser (pip install ripser) as well as persistable from the higher-homology branch,
# which can be done as follows:
#     pip install git+https://github.com/LuisScoccola/persistable.git@higher-homology
# NOTE: only accepts as input a distance matrix
def hf_degree_rips(
    distance_matrix,
    min_rips_value,
    max_rips_value,
    max_normalized_degree,
    min_normalized_degree,
    grid_granularity,
    max_homological_dimension,
    subsample_size = None,
):
    if subsample_size == None:
        p = persistable.Persistable(distance_matrix, metric="precomputed")
    else:
        p = persistable.Persistable(distance_matrix, metric="precomputed", subsample=subsample_size)

    rips_values, normalized_degree_values, hilbert_functions, minimal_hilbert_decompositions = p._hilbert_function(
        min_rips_value,
        max_rips_value,
        max_normalized_degree,
        min_normalized_degree,
        grid_granularity,
        homological_dimension=max_homological_dimension,
    )

    return rips_values, normalized_degree_values, hilbert_functions, minimal_hilbert_decompositions



def hf_h0_degree_rips(
    point_cloud,
    min_rips_value,
    max_rips_value,
    max_normalized_degree,
    min_normalized_degree,
    grid_granularity,
):
    p = persistable.Persistable(point_cloud, n_neighbors="all")

    rips_values, normalized_degree_values, hilbert_functions, minimal_hilbert_decompositions = p._hilbert_function(
        min_rips_value,
        max_rips_value,
        max_normalized_degree,
        min_normalized_degree,
        grid_granularity,
    )

    return rips_values, normalized_degree_values, hilbert_functions[0], minimal_hilbert_decompositions[0] 


def ri_h0_degree_rips(
    point_cloud,
    min_rips_value,
    max_rips_value,
    max_normalized_degree,
    min_normalized_degree,
    grid_granularity,
):
    p = persistable.Persistable(point_cloud, n_neighbors="all")

    rips_values, normalized_degree_values, rank_invariant, _, _ = p._rank_invariant(
        min_rips_value,
        max_rips_value,
        max_normalized_degree,
        min_normalized_degree,
        grid_granularity,
    )

    return rips_values, normalized_degree_values, rank_invariant




