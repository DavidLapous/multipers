// todo avec tensioru

#pragma once

#include <iostream>
#include <vector>
#include <utility>  // std::pair
#include <tuple>
#include <iterator>  // for std::distance
#include <numeric>
#include <algorithm>
#include "Simplex_tree_multi_interface.h"
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include "tensor/tensor.h"
#include "multi_parameter_rank_invariant/persistence_slices.h"


namespace Gudhi::multiparameter::rank_invariant {


// using Elbow = std::vector<std::pair<>>;
template<typename index_type>
inline void push_in_elbow(index_type& i,index_type& j, const index_type I, const index_type J){ 	
	if (j<J) {j++;return;}
	if (i<I) {i++;return;}
	j++; return;
}
template<typename index_type>
inline value_type get_slice_rank_filtration(const value_type x,const value_type y,const index_type I, const index_type J){
	if (x>I)	return std::numeric_limits<value_type>::infinity();
	if (y>J)	return I+static_cast<index_type>(y);
	// if (x>0)	//not necessary
	return J+static_cast<index_type>(x);
	// return J;
}

template<typename index_type>
inline std::pair<index_type,index_type> get_coordinates(index_type in_slice_value, index_type I, index_type J){
	if (in_slice_value <= J )	return {0,J};
	if (in_slice_value <= I+J) 	return {in_slice_value-J,J};
	return {I, in_slice_value - I};
}


template<typename dtype, typename index_type>
inline void compute_2d_rank_invariant_of_elbow(
	Simplex_tree_multi &st_multi,
	Simplex_tree_std &_st_container, // copy of st_multi
	const tensor::static_tensor_view<dtype, index_type>& out, // assumes its a zero tensor
	const index_type I, const index_type J,
	const std::vector<index_type>& grid_shape,
	const std::vector<index_type>& degrees,
	const int expand_collapse_max_dim=0
){

	// const auto X = grid_shape[1],  Y = grid_shape[2]; // First axis is degree 
	const auto Y = grid_shape[2];
	// Fills the filtration in the container
	// TODO : C++23 zip, when Apples clang will stop being obsolete
	auto sh_standard = _st_container.complex_simplex_range().begin();
	auto _end = _st_container.complex_simplex_range().end();
	auto sh_multi = st_multi.complex_simplex_range().begin();
	for (;sh_standard != _end; ++sh_multi, ++sh_standard){
		const auto& multi_filtration = st_multi.filtration(*sh_multi);
		index_type x = static_cast<index_type>(multi_filtration[0]);
		index_type y = static_cast<index_type>(multi_filtration[1]);
		auto filtration_in_slice = get_slice_rank_filtration(x,y,I,J);
		_st_container.assign_filtration(*sh_standard, filtration_in_slice);
	}
	const std::vector<Barcode>& barcodes = compute_dgms(_st_container, degrees,expand_collapse_max_dim);
	index_type degree_index=0;
	for (const auto& barcode : barcodes){ // TODO range view cartesian product
		for (const auto &bar : barcode){
			auto birth = static_cast<index_type>(bar.first); 
			auto death = static_cast<index_type>(std::min(bar.second,static_cast<value_type>(Y+I))); // I,J atteints, pas X ni Y

			// todo : optimize
			// auto [a,b] = get_coordinates(birth, I,J);
			for (auto intermediate_birth=birth; intermediate_birth<death;intermediate_birth++){
				for (auto intermediate_death=intermediate_birth; intermediate_death<death;intermediate_death++){
					auto [i,j] = get_coordinates(intermediate_birth, I,J);
					auto [k,l] = get_coordinates(intermediate_death, I,J);
					if (((i < k || j == J) && (j < l || k == I) )){ 
						// std::vector<index_type> coordinates_to_remove = {degree_index,i,j,k,l};
						// out[coordinates_to_remove]++;
						out[{degree_index,i,j,k,l}]++;
					}
				}
			}
		}
		degree_index++;
	}
}

template<typename dtype, typename index_type>
inline void compute_2d_rank_invariant(
	Simplex_tree_multi &st_multi,
	const tensor::static_tensor_view<dtype, index_type>& out, // assumes its a zero tensor
	const std::vector<index_type>& grid_shape,
	const std::vector<index_type>& degrees,
	bool expand_collapse
){
	if (degrees.size() == 0) return;
	assert(st_multi.get_number_of_parameters() == 2);
	Simplex_tree_std st_;
	flatten(st_, st_multi,0); // copies the st_multi to a standard 1-pers simplextree
	const int max_dim = expand_collapse ? *std::max_element(degrees.begin(), degrees.end()) +1 : 0;
	index_type X = grid_shape[1]; index_type Y = grid_shape[2]; // First axis is degree 
	tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree(st_); // initialize with a good simplextree
	tbb::parallel_for(0, X,[&](index_type I){
		tbb::parallel_for(0,Y, [&](index_type J){
			auto& st_container = thread_simplex_tree.local();
			compute_2d_rank_invariant_of_elbow(
				st_multi,st_container,out,I,J,grid_shape,degrees, max_dim
			);
		});
	});

}



template<typename dtype, typename indices_type, typename ... Args>
void compute_rank_invariant_python(
	const std::intptr_t simplextree_ptr, 
	dtype* data_ptr, 
	const std::vector<indices_type> grid_shape,
	const std::vector<indices_type> degrees,
	indices_type n_jobs,
	bool expand_collapse
	){
	if (degrees.size() == 0) return;
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	tensor::static_tensor_view<dtype, indices_type> container(data_ptr,grid_shape); // assumes its a zero tensor

	oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
	arena.execute([&]{
		compute_2d_rank_invariant(st_multi,container, grid_shape, degrees, expand_collapse);
	});

	return;
}

}
