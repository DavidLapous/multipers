/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2014 Inria        """
 *
 *
 *    Modification(s):
 * 		- 2022/11 David Loiseaux / Hannah Schreiber : added multify / flatten to interface standard simplextree.
 *      - YYYY/MM Author: Description of the modification
 */
#ifndef SIMPLEX_TREE_MULTI_H_
#define SIMPLEX_TREE_MULTI_H_

#include <algorithm>
#include "gudhi/Simplex_tree.h"

namespace Gudhi {

/** Model of SimplexTreeOptions.
 *
 * Maximum number of simplices to compute persistence is <CODE>std::numeric_limits<std::uint32_t>::max()</CODE>
 * (about 4 billions of simplices). */
struct Simplex_tree_options_multidimensional_filtration {
	typedef linear_indexing_tag Indexing_tag;
	typedef int Vertex_handle;
	typedef std::vector<double> Filtration_value;
	typedef std::uint32_t Simplex_key;
	static const bool store_key = true;
	static const bool store_filtration = true;
	static const bool contiguous_vertices = false;
};
using option_multi = Simplex_tree_options_multidimensional_filtration;
using option_std = Simplex_tree_options_full_featured;
bool operator<(const std::vector<double>& v1, const std::vector<double>& v2)
{
	bool isSame = true;
	if (v1.size() != v2.size()) isSame = false;
	for (unsigned int i = 0; i < std::min(v1.size(), v2.size()); ++i){
		if (v1[i] > v2[i]) return false;
		if (isSame && v1[i] != v2[i]) isSame = false;
	}
	if (isSame) return false;
	return true;
}

void multify(const uintptr_t splxptr, const uintptr_t newsplxptr, const unsigned int dimension = 1){
	Simplex_tree<option_std> &st = *(Gudhi::Simplex_tree<option_std>*)(splxptr);
	Simplex_tree<option_multi> &st_multi = *(Gudhi::Simplex_tree<option_multi>*)(newsplxptr);;
	if (dimension <= 0)
		{std::cout << "Empty filtration\n"; return ;}
	std::vector<double> f(dimension);
	for (auto &simplex_handle : st.complex_simplex_range()){
		std::vector<int> simplex;
		for (auto vertex : st.simplex_vertex_range(simplex_handle))
			simplex.push_back(vertex);
		f[0] = st.filtration(simplex_handle);
		st_multi.insert_simplex(simplex,f);
	}
}
void flatten(const uintptr_t splxptr, const uintptr_t newsplxptr, const unsigned int dimension = 0){
	Simplex_tree<option_std> &st = *(Gudhi::Simplex_tree<option_std>*)(newsplxptr);
	Simplex_tree<option_multi> &st_multi = *(Gudhi::Simplex_tree<option_multi>*)(splxptr);

	for (const auto &simplex_handle : st_multi.complex_simplex_range()){
		std::vector<int> simplex;
		for (auto vertex : st_multi.simplex_vertex_range(simplex_handle))
			simplex.push_back(vertex);
		double f = st_multi.filtration(simplex_handle)[dimension];
		st.insert_simplex(simplex,f);
	}
}

}	// namespace Gudhi

namespace std {

template<>
class numeric_limits<std::vector<double> >
{
public:
	static std::vector<double> infinity() throw(){
		return std::vector<double>(1, numeric_limits<double>::infinity());
	};


	static std::vector<double> quiet_NaN() throw(){
		return std::vector<double>(1, numeric_limits<double>::quiet_NaN());
	};

};

}	// namespace std





#endif  // SIMPLEX_TREE_MULTI_H_
