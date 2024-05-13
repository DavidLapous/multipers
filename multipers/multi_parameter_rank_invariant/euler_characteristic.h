#pragma once

#include <iostream>
#include <vector>
#include <utility>  // std::pair
#include "Simplex_tree_multi_interface.h"
#include "multi_parameter_rank_invariant/persistence_slices.h"

#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>

#include "tensor/tensor.h"


namespace Gudhi::multiparameter::euler_characteristic{


template<typename Filtration, typename dtype=int, typename index_type=std::uint16_t>
void get_euler_surface(
	python_interface::Simplex_tree_multi_interface<Filtration, typename Filtration::value_type> &st_multi,
	const tensor::static_tensor_view<dtype, index_type>& out, // assumes its a zero tensor
	bool mobius_inversion,
	bool zero_pad
	){
	std::vector<index_type> coordinate_container(st_multi.get_number_of_parameters()); 
	for (auto sh : st_multi.complex_simplex_range()){
		const auto& multi_filtration = st_multi.filtration(sh);
		for (index_type i=0u; i<st_multi.get_number_of_parameters(); i++){
			coordinate_container[i] = static_cast<index_type>(multi_filtration[i]);
		}
		int sign = 1-2*(st_multi.dimension(sh) % 2);
		if (mobius_inversion && zero_pad)
			out.add_cone_boundary(coordinate_container,sign);
		else if (mobius_inversion)
			out[coordinate_container] += sign;
		else
			out.add_cone(coordinate_container,sign);
	}
	return;
}


template<typename Filtration, typename dtype=int, typename indices_type=uint16_t>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> get_euler_signed_measure(
	python_interface::Simplex_tree_multi_interface<Filtration, typename Filtration::value_type>& st_multi, 
	dtype* data_ptr, 
	std::vector<indices_type> grid_shape,
	bool zero_pad,
	const bool verbose = false){
	// const bool verbose = false;
	tensor::static_tensor_view<dtype, indices_type> container(data_ptr,grid_shape); // assumes its a zero tensor
	if (verbose){
		std::cout << "Container shape : ";
		for (auto r : container.get_resolution()) std::cout << r << ", ";
		std::cout << "\nContainer size : " << container.size();
		std::cout << "\nComputing Euler Characteristic ...";
	}
	get_euler_surface(st_multi,container,true, zero_pad);
	if (verbose){
		std::cout << "Done." << std::endl;
		std::cout << "Sparsifying the measure ...";
	}
	auto raw_signed_measure = container.sparsify();
	if (verbose){
		std::cout << "Done." << std::endl;
	}
	return raw_signed_measure;
}


template<typename Filtration, typename dtype, typename indices_type, typename ... Args>
void get_euler_surface_python(
	const intptr_t simplextree_ptr, 
	dtype* data_ptr, 
	const std::vector<indices_type> grid_shape,
	bool mobius_inversion=false, 
	bool zero_pad = false, 
	bool verbose=false){
	auto &st_multi = get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
	tensor::static_tensor_view<dtype, indices_type> container(data_ptr,grid_shape); // assumes its a zero tensor
	if (verbose){
		std::cout << "Container shape : ";
		for (auto r : container.get_resolution()) std::cout << r << ", ";
		std::cout << "\nContainer size : " << container.size();
		std::cout << "\nComputing Euler Characteristic ...";
	}
	get_euler_surface(st_multi,container,mobius_inversion, zero_pad);
	if (verbose){
		std::cout << "Done." << std::endl;
	}
	return;
}





} // namespace rank_invariant
