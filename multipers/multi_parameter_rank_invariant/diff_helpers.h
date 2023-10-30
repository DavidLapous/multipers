#pragma once

#include <iostream>
#include <vector>
#include "Simplex_tree_multi_interface.h"


namespace Gudhi::multiparameter::differentiation{

using value_type = Simplex_tree_options_multidimensional_filtration::Filtration_value::value_type;
using signed_measure_indices = std::vector<std::vector<int32_t>>  ;
using Simplex_tree_multi = interface_multi;
using signed_measure_pts = std::vector<std::vector<value_type>>;



std::vector<std::map<value_type, int32_t>> build_idx_map(
	Simplex_tree_multi& st, 
	const std::vector<int>& simplices_dimensions
){
	auto num_parameters = st.get_number_of_parameters();
	if (static_cast<int>(simplices_dimensions.size()) < num_parameters) throw;
	int max_dim = *std::max_element(simplices_dimensions.begin(),simplices_dimensions.end());
	int min_dim = *std::min_element(simplices_dimensions.begin(),simplices_dimensions.end());
	max_dim = min_dim >= 0 ? max_dim : st.dimension();

	std::vector<std::map<value_type, int32_t>> idx_map(num_parameters);
	auto splx_idx = 0u;
	for (auto sh : st.complex_simplex_range()){ // order has to be retrieved later, so I'm not sure that skeleton iterator is well suited
		const auto& splx_filtration = st.filtration(sh);
		const auto splx_dim = st.dimension(sh);
		if (splx_dim<=max_dim)
			for (auto i=0u;i<splx_filtration.size();i++){
				if (simplices_dimensions[i] != splx_dim and simplices_dimensions[i] != -1)	continue;
				auto f = splx_filtration[i];
				idx_map[i].try_emplace(f, splx_idx);
			}
		splx_idx++;
	}
	return idx_map;
}
template<typename ... Args>
std::vector<std::map<value_type, int32_t>> build_idx_map(const intptr_t simplextree_ptr,Args...args){
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	return build_idx_map(st_multi,args...);
}



std::pair<signed_measure_indices, signed_measure_indices> get_signed_measure_indices(
		const std::vector<std::map<value_type, int32_t>>& idx_map, 
		const signed_measure_pts& pts
	){
	auto num_pts = pts.size();
	auto num_parameters = idx_map.size();
	signed_measure_indices out_indices(num_pts, std::vector<int32_t>(num_parameters, -1)); // -1 to be able from indicies to get if the pt is found or not
	signed_measure_indices out_unmapped_values;
	for (auto pt_idx =0u; pt_idx < num_pts; pt_idx++){
		auto& pt = pts[pt_idx];
		auto& pt_indices = out_indices[pt_idx];

		for (auto parameter=0u;parameter<num_parameters;parameter++){
			value_type f = pt[parameter];
			const std::map<value_type, int32_t>& parameter_map = idx_map[parameter];
			auto it = parameter_map.find(f);
			if (it == parameter_map.end())
				out_unmapped_values.push_back({static_cast<int32_t>(pt_idx),static_cast<int32_t>(parameter)});
			else
				pt_indices[parameter] = it->second;
		}
	}
	return {out_indices, out_unmapped_values}; // TODO return a ptr for python 
}




} // namespace Gudhi::multi::differentiation