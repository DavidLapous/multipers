#pragma once

#include <iostream>
#include <vector>
#include <utility>  // std::pair
#include <tuple>
#include <iterator>  // for std::distance
#include <numeric>
#include <algorithm>
#include "gudhi/Simplex_tree_multi_interface.h"
#include "multi_filtrations/box.h"
#include "multi_filtrations/finitely_critical_filtrations.h"
#include "multi_filtrations/line.h"
//#include "temp/debug.h"
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/global_control.h>
#include <ranges>
#include "tensor/tensor.h"
#include "multi_parameter_rank_invariant/persistence_slices.h"

// somewhere



namespace Gudhi::multiparameter::rank_invariant{


using signed_measure = std::pair< std::vector<std::vector<int>>, std::vector<int>>;
using grid1d = std::vector<int>;
using grid2d = std::vector<grid1d>;
using grid3d = std::vector<grid2d>;
using grid4d = std::vector<grid3d>;
using grid5d = std::vector<grid4d>;






template<typename T=int>
using Rectangle = std::tuple<std::vector<T>, std::vector<T>, int>;
inline grid2d allocate_zero_grid(int a, int b){
	grid2d out(a, std::vector<int>(b,0));
	return out;
}
inline grid3d allocate_zero_grid(int a, int b,int c){
	grid3d out(a, allocate_zero_grid(b,c));
	return out;
}
inline grid4d allocate_zero_grid(int a, int b,int c, int d){
	grid4d out(a, allocate_zero_grid(b,c,d));
	return out;
}





inline void möbius_inversion(std::vector<int>& x, bool zero_pad){
	// const int n = x.size();
	// is_in_range = [n](int i){return i>= 0 && i < n;};
	if (zero_pad)
		x.back() = 0;
	int a=0,b=0;
	for (unsigned int i = 0; i<x.size(); i++){
		a = b;
		b = x[i];
		x[i] = b-a;
	}
}
inline void möbius_inversion(const std::vector<int*>& x, int pointer_shift, bool zero_pad){
	// const int n = x.size();
	// is_in_range = [n](int i){return i>= 0 && i < n;};
	constexpr bool verbose = false;
	int a=0,b=0;
	if (zero_pad)
		*(x.back()+pointer_shift) = 0;
	for (unsigned int i = 0; i < x.size(); i++){
		a = i == 0 ? 0 : b;
		b = *(x[i]+pointer_shift);
		if constexpr (verbose)	std::cout << "Line "<< i << " old " << a << " new " << b << "\n";
		*(x[i]+pointer_shift) = b-a;
	}
}

inline void möbius_inversion2d(const std::vector<int*>& x, int max_pointer_shift, bool zero_pad){
	tbb::parallel_for(
		0,max_pointer_shift,[&](int j){ // x is assumed to be a matrix
			möbius_inversion(x, j, zero_pad);
		}
	);
}

inline void möbius_inversion3d(const std::vector<std::vector<int*>>& x, int max_pointer_shift, bool zero_pad){
	tbb::parallel_for(
		0,static_cast<int>(x.size()),[&](int j){
			möbius_inversion(x[j], max_pointer_shift, zero_pad);
		}
	);
}


inline void möbius_inversion(grid2d& x, bool zero_pad, int axis = -1){
	// Last axis mobius inversion can be done when storing the bars

	// const int n = x.size();
	// is_in_range = [n](int i){return i>= 0 && i < n;};
	if (axis == 1 or axis < 0){
		tbb::parallel_for(
			0,static_cast<int>(x.size()),[&](int i){
				möbius_inversion(x[i], zero_pad);
			}
		);
	}
	// if (zero_pad){ // axis 0 is already inverted, only need to zero_pad if necessary
	// 	for( auto i= 0u; i<x.size();i++){
	// 		x[i].back() = - std::accumulate(x[i].begin(), x[i].end()-1,0);
	// 	}
	// }


	// if (axis == 0 or axis < 0){
	// 	// if (zero_pad){
	// 	// 	x.back() = std::vector<int>(x[0].size(),0);
	// 	// }
	// 	std::vector<int*> row_pointers(x.size());
	// 	for (unsigned int i=0; i < x.size();i++)
	// 		row_pointers[i] = &x[i][0];
	// 	tbb::parallel_for(
	// 		0,static_cast<int>(x[0].size()),[&](int j){ // x is assumed to be a matrix
	// 			möbius_inversion(row_pointers, j, zero_pad);
	// 		}
	// 	);
	// }
}


inline void möbius_inversion(grid3d& x, bool zero_pad){
	// axes 1 and 2
	tbb::parallel_for(
		0,static_cast<int>(x.size()),[&](int i){
			möbius_inversion(x[i], zero_pad);
		}
	);
	// axe 0
	// if (zero_pad)
	// 	x.back() = allocate_zero_grid(x[0].size(), x[0][0].size());
	
	std::vector<int*> row_pointers(x.size());
	for (unsigned int j = 0; j< x[0].size(); j++){
		for (unsigned int i=0; i < x.size();i++){
			row_pointers[i] = &x[i][j][0];
		}
		for (unsigned int k=0; k<x[0][0].size();k++)
			möbius_inversion(row_pointers,k, zero_pad);
	}
}

inline void möbius_inversion(grid4d& x, bool zero_pad){
	// axes 1, 2, and 3
	tbb::parallel_for(
		0,static_cast<int>(x.size()),[&](int i){
			möbius_inversion(x[i], zero_pad);
		}
	);
	// axe 0
	if (zero_pad)
		x.back() = allocate_zero_grid(x[0].size(), x[0][0].size(), x[0][0][0].size());
	
	std::vector<int*> row_pointers(x.size());
	for (unsigned int j = 0; j< x[0].size(); j++){
		for (unsigned int k=0; k<x[0][0].size();k++){
			for (unsigned int i=0; i < x.size();i++){
				row_pointers[i] = &x[i][j][k][0];
			}
			for (unsigned int l=0; l<x[0][0][0].size();l++)
				möbius_inversion(row_pointers,l, zero_pad);
		}
	}
}


inline signed_measure sparsify(const grid2d& tensor){
	signed_measure out;
	auto& pts = out.first;
	auto& weights = out.second;
	for (int i=0; i < static_cast<int>(tensor.size()); i++){
		for (int j=0; j < static_cast<int>(tensor[0].size()); j++){
			if (tensor[i][j] != 0){
				pts.push_back({i,j});
				weights.push_back(tensor[i][j]);
			}
		}
	}
	return out;
}

inline signed_measure sparsify(const grid3d& tensor){
	signed_measure out;
	auto& pts = out.first;
	auto& weights = out.second;
	for (int i=0; i < static_cast<int>(tensor.size()); i++){
		for (int j=0; j < static_cast<int>(tensor[0].size()); j++){
			for (int k=0; k < static_cast<int>(tensor[0][0].size()); k++){
				if (tensor[i][j][k] != 0){
					pts.push_back({i,j,k});
					weights.push_back(tensor[i][j][k]);
				}
			}
		}
	}
	return out;
}

inline signed_measure sparsify(const grid4d& tensor){
	signed_measure out;
	auto& pts = out.first;
	auto& weights = out.second;
	for (int i=0; i < static_cast<int>(tensor.size()); i++){
		for (int j=0; j < static_cast<int>(tensor[0].size()); j++){
			for (int k=0; k < static_cast<int>(tensor[0][0].size()); k++){
				for (int l=0;l< static_cast<int>(tensor[0][0][0].size()); l++){
					if (tensor[i][j][k][l] != 0){
						pts.push_back({i,j,k,l});
						weights.push_back(tensor[i][j][k][l]);
					}
				}
			}
		}
	}
	return out;
}





inline void project_to_elbow(std::vector<value_type> &to_project, value_type i, value_type j, value_type I, value_type J){
	// Box<value_type> top_left_zone(0,j+1, i-1,J-1);
	// Box<value_type> bottom_left_zone(1,0, i,j-1);
	// Box<value_type> right_zone(i+1,0,I-1,J-1);
	Box<value_type> zone(1,0, i,j-1); // Bottom left zone
	auto &birth = zone.get_bottom_corner();
	auto &death = zone.get_upper_corner();
	if (zone.contains(to_project)){
		to_project[1] = j;
		// projection = {x[0], j};
		return;
	}
	birth[0] = 0; birth[1] = j+1; death[0] = i-1; death[1] = J-1;
	if (zone.contains(to_project)) //top left zone
		{to_project[0] = i; return;}
		// projection = {i,x[1]};
	birth[0] = i+1; birth[1] = 0; death[0] = I-1; //death[1] = J-1;
	if (zone.contains(to_project)) //right zone
		{to_project[1] = J-1; return;}
		// projection = {x[0], J-1};
	return;
	// // if (close_top && projection[1] == j){
	// // 	projection[1] = j+0.1;
	// // }
	// return to_project;
}




using Elbow = std::vector<std::vector<int>>;
inline Elbow get_elbow(int i,int j,int I, int J){ 
	constexpr bool verbose = false;
	std::vector<std::vector<int>> out(I+J, std::vector<int>(2));
	if constexpr (verbose) std::cout << "Computing elbow " << i << " " << j << std::endl;
	int _i=0, _j=0;
	while (_j < j){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_j++;
	}
	while(_i < i){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_i++;
	}
	while (_j < J){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_j++;
	}
	_j--;
	_i++;
	while(_i < I){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_i++;
	}
	out[I+J-1] = {I,J};
	return out;
}


// For 2_dimensional rank
using rank_tensor = std::vector<std::vector<std::vector<std::vector<int>>>>;
// assumes that the simplextree has grid coordinate filtration
rank_tensor get_2drank_invariant(const intptr_t simplextree_ptr, const std::vector<int> &grid_shape, const int degree){
	constexpr bool verbose=false;

	Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
	int I = grid_shape[0], J = grid_shape[1];
	rank_tensor out(I, std::vector<std::vector<std::vector<int>>>(
					J, std::vector<std::vector<int>>(
					I, std::vector<int>(
					J,0)))); // zero of good size
	// std::cout << I <<" " << J << std::endl;
	Simplex_tree_std st_;
	flatten(st_, st_multi,0); // copies the st_multi to a standard 1-pers simplextree

	tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree;
	tbb::parallel_for(0, I,[&](int i){
		tbb::parallel_for(0,J, [&](int j){
			// gets the thread local variables
			Simplex_tree_std &st = thread_simplex_tree.local();
			const Elbow &elbow_container = get_elbow(i,j,I,J);
			if (st.num_simplices() == 0){ st = st_;}
			if constexpr (verbose) std::cout <<"\nElbow : "<<  i << " " << j << std::endl;

			Simplex_tree_multi::Filtration_value multi_filtration;
			auto sh_multi = st_multi.complex_simplex_range().begin(); // relies on the fact that this iterator is determinstic for two simplextrees having the same simplices
			auto sh_standard = st.complex_simplex_range().begin();
			auto _end = st.complex_simplex_range().end();
			for (; sh_standard != _end; ++sh_standard, ++sh_multi){
				multi_filtration = st_multi.filtration(*sh_multi);
				project_to_elbow(multi_filtration,i ,j, I,J);
				auto elbow_filtration = multi_filtration[0] + multi_filtration[1];
				st.assign_filtration(*sh_standard, elbow_filtration);
			}
			
			const Barcode barcode = compute_dgm(st, degree);
			for(const auto &bar : barcode){
				int birth = static_cast<int>(bar.first);
				int death = bar.second == std::numeric_limits<int>::infinity() ? I+J-1: static_cast<int>(bar.second); // TODO FIXME 
				
				//Thresholds:
				if constexpr (verbose) std::cout <<"Bar " << birth << " " << death << std::endl;
				birth = std::max<int>(birth, j);
				death = std::min<int>(death, J-1 + i);
				if constexpr (verbose) std::cout <<"Thresholded Bar " << birth << " " << death << std::endl;

				for (int b = birth; b < death; b ++){
					for(int d = b; d < death; d++ ){
						const std::vector<int> &birth_coordinates = elbow_container[b];
						const std::vector<int> &death_coordinates = elbow_container[d];
						
						int b1 = birth_coordinates[0], b2 = birth_coordinates[1];
						int d1 = death_coordinates[0], d2 = death_coordinates[1];
						if ((b1 != d1 || b2 == j) && (b2 != d2 || d1 == i)){
							out[b1][b2][d1][d2]++;
						}
					}
				}
			}
			

		});
	});

	return out;
}






/// @brief Project a filtration value to an horizontal line.
/// @param x the value to project
/// @param height 
/// @param i 
/// @param j 
/// @param fixed_values 
/// @return
template<typename fixed_values_type>
inline value_type horizontal_line_filtration(const std::vector<value_type> &x, value_type height, int i, int j, const std::vector<fixed_values_type>& fixed_values){
	for (int k = 0, count = -1; k < static_cast<int>(x.size()); k++){
		if (k == i || k == j) continue; // coordinate in the plane
		count++;
		if (x[k] > fixed_values[count]) // simplex appears after the plane
			return std::numeric_limits<Simplex_tree_std::Filtration_value>::infinity();
	}
	if (x[j] <= height) // simplex apppears in the plane, but is it in the line with height "height"
		return x[i];
	else
		return std::numeric_limits<Simplex_tree_std::Filtration_value>::infinity();
}

// inline assign_std_simplextree_from_multi(Simplex_tree_std& st,const Simplex_tree_multi& st_multi, function_type)





// inline assign_std_simplextree_from_multi(Simplex_tree_std& st,const Simplex_tree_multi& st_multi, function_type)







/// @brief Computes the hilbert function on a 2D grid. It assumes that the filtration values of the simplextree are coordinates in a grid, of size grid_shape
/// @param st_multi the multiparameter simplextree. It has to have at least 2 parameters. i,j are the filtration axes on which to compute the filtration
/// @param grid_shape the size of the grid
/// @param degree the homological degree to compute
/// @param i free coordinate 
/// @param j free coordinate;
/// @param fixed_values when the simplextree is more than 2 parameter, the non-free coordinate have to be specified, i.e. on which "plane" to compute the hilbert function.
/// @return the hilbert function
grid3d get_2Dhilbert(Simplex_tree_multi &st_multi, const std::vector<int> grid_shape, const std::vector<int> degrees, int i = 0, int j = 1, const std::vector<value_type> fixed_values = {}, bool mobius_inverion = false, bool zero_pad=false){
	if (grid_shape.size() < 2 || st_multi.get_number_of_parameters() < 2)
		throw std::invalid_argument("Grid shape has to have at least 2 element.");
	if (st_multi.get_number_of_parameters() - fixed_values.size() != 2)
		throw std::invalid_argument("Fix more values for the simplextree, which has a too big number of parameters");
	constexpr bool verbose = false;
	if constexpr(verbose)
		tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
	int I = grid_shape[i], J = grid_shape[j];
	if constexpr(verbose) std::cout << "Grid shape : " << I << " " << J << std::endl;

	// grid2d out(I, std::vector<int>(J,0)); // zero of good size
	grid3d out = allocate_zero_grid(degrees.size(),I,J);
	Simplex_tree_std _st;
	flatten(_st, st_multi,-1); // copies the st_multi to a standard 1-pers simplextree
	tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree;
	tbb::parallel_for(0, J,[&](int height){
		Simplex_tree_std &st_std = thread_simplex_tree.local();
		if (st_std.num_simplices() == 0){ st_std = _st;}
		Simplex_tree_multi::Filtration_value multi_filtration;
		auto sh_standard = st_std.complex_simplex_range().begin();
		auto _end = st_std.complex_simplex_range().end();
		auto sh_multi = st_multi.complex_simplex_range().begin();
		for (;sh_standard != _end; ++sh_multi, ++sh_standard){
			multi_filtration = st_multi.filtration(*sh_multi); 
			value_type horizontal_filtration = horizontal_line_filtration(multi_filtration, height, i,j, fixed_values);
			st_std.assign_filtration(*sh_standard, horizontal_filtration);
			if constexpr (verbose){
				Simplex_tree_multi::Filtration_value splx;
				for (auto vertex : st_multi.simplex_vertex_range(*sh_multi))	splx.push_back(vertex);
				std::cout << "Simplex " << splx << "/"<< st_std.num_simplices() << " Filtration multi " << st_multi.filtration(*sh_multi) << " Filtration 1d " <<  st_std.filtration(*sh_standard) << "\n";
			}
		}
		if constexpr(verbose) {
			std::cout << "Coords : "  << height << " [";
			for (auto stuff : fixed_values)
				std::cout << stuff << " ";
			std::cout  << "]" << std::endl;
		}
		const std::vector<Barcode> barcodes = compute_dgms(st_std, degrees);
		int degree_index=0;
		for (const auto& barcode : barcodes){
			for(const auto &bar : barcode){
				auto birth = bar.first;
				auto death = bar.second;
				// if constexpr (verbose) std::cout << "BEFORE " << birth << " " << death << " " << I << " \n";
				// death = death > I ? I : death; // TODO FIXME 
				// if constexpr (verbose) std::cout <<"AFTER" << birth << " " << death << " " << I << " \n";
				if (birth > I) // some birth can be infinite
					continue;
				
				if (!mobius_inverion){
					death = death > I ? I : death;
					for (int index = static_cast<int>(birth); index < static_cast<int>(death); index ++){
						out[degree_index][index][height]++;
					}
				}
				else{
					out[degree_index][static_cast<int>(birth)][height]++; // No need to do mobius inversion on this axis, it can be done here
					if (death < I)
						out[degree_index][static_cast<int>(death)][height]--;
					else if (zero_pad)
					{
						out[degree_index].back()[height]--;
					}
					
				}
				// else 
				// 	out[I-1][height]--;
			}
			degree_index++;
		}

	});
	return out;
}



/// @brief /!\ DANGEROUS /!\ For python only. 
/// @tparam ...Args 
/// @param simplextree_ptr the simplextree pointer
/// @param ...args 
/// @return 
template<typename ... Args>
grid3d get_2Dhilbert(const intptr_t simplextree_ptr, Args...args){
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	return get_2Dhilbert(st_multi, args...);
}

/// @brief 
/// @param st_multi simplextree 
/// @param grid_shape shape of the 3D grid
/// @param degree homological degree to compute
/// @param i free coordinate
/// @param j free coordinate
/// @param k free coordinate
/// @param fixed_values values of the non-free coordinates
/// @return 
grid4d get_3Dhilbert(Simplex_tree_multi &st_multi, const std::vector<int> grid_shape, const std::vector<int> degrees, int i=0, int j=1, int k=2,const std::vector<value_type> fixed_values = {}, bool mobius_inverion = false, bool zero_pad=false){
	if (grid_shape.size() < 3 || st_multi.get_number_of_parameters() < 3 )
		throw std::invalid_argument("Grid shape has to have at least 3 element.");
	if (st_multi.get_number_of_parameters() - fixed_values.size() != 3)
		throw std::invalid_argument("Fix more values for the simplextree, which has a too big number of parameters");
	grid4d out(degrees.size(), grid3d(grid_shape[i]));
	// const std::vector<int> _grid = {grid_shape[1],grid_shape[2]};
	tbb::parallel_for(0, static_cast<int>(grid_shape[i]), [&](int z){
		std::vector<value_type> _fixed_values(fixed_values.size() +1);
		_fixed_values[0] = static_cast<value_type>(z);
		std::copy(fixed_values.begin(), fixed_values.end(), _fixed_values.begin()+1);
		const auto& slice = get_2Dhilbert(st_multi, grid_shape, degrees, j,k, _fixed_values, mobius_inverion, zero_pad);
		for (auto degree_index = 0u; degree_index < degrees.size(); degree_index++){
			out[degree_index][z] = slice[degree_index];
		}
	});
	return out;
}
template<typename ... Args>
/// @brief /!\ DANGEROUS /!\ For python only. 
/// @tparam ...Args 
/// @param simplextree_ptr the simplextree pointer
/// @param ...args 
/// @return 
grid4d get_3Dhilbert(const intptr_t simplextree_ptr, Args...args){
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	return get_3Dhilbert(st_multi, args...);
}

grid5d get_4Dhilbert(Simplex_tree_multi &st_multi, const std::vector<int> grid_shape, const std::vector<int> degrees, const std::vector<value_type> fixed_values = {}, bool mobius_inverion = false, bool zero_pad=false){
	if (grid_shape.size() < 4 || st_multi.get_number_of_parameters() < 4)
		throw std::invalid_argument("Grid shape has to have at least 4 element.");
	if (st_multi.get_number_of_parameters() - fixed_values.size() != 4)
		throw std::invalid_argument("Fix more values for the simplextree, which has a too big number of parameters");
	
	grid5d out(degrees.size(), grid4d(grid_shape[0]));
	// const std::vector<int> _grid = {grid_shape[1],grid_shape[2], grid_shape[3]};
	tbb::parallel_for(0, static_cast<int>(grid_shape[0]), [&](int z){
		// out[z] = get_3Dhilbert(st_multi, grid_shape, degree, );
		auto slice = get_3Dhilbert(st_multi, grid_shape, degrees, 1,2,3, {static_cast<value_type>(z)}, mobius_inverion, zero_pad);
		for (auto degree_index = 0u; degree_index < degrees.size(); degree_index++){
			out[degree_index][z] = slice[degree_index];
		}
	});
	return out;
}

template<typename ... Args>
/// @brief /!\ DANGEROUS /!\ For python only. 
/// @tparam ...Args 
/// @param simplextree_ptr the simplextree pointer
/// @param ...args 
/// @return 
grid5d get_4Dhilbert(const intptr_t simplextree_ptr, Args...args){
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	return get_4Dhilbert(st_multi, args...);
}

inline void add_above(std::vector<int>& x, options_multi::value_type threshold, int value){
	for (unsigned int i = static_cast<unsigned int>(threshold); i < x.size(); i++){
		x[i]+=value;
	}
}

inline void add_above(grid2d& x, const options_multi::Filtration_value& threshold, int value){
	for (unsigned int i = threshold[0]; i < x.size(); i++){
		add_above(x[i], static_cast<int>(threshold[1]), value);
	}
}
inline void add_above(grid3d& x, const options_multi::Filtration_value& threshold, int value){
	const options_multi::Filtration_value sub_threshold(threshold.begin()+1, threshold.end());
	for (unsigned int i = threshold[0]; i < x.size(); i++){
		add_above(x[i], sub_threshold, value);
	}
}
inline void add_above(grid4d& x, const options_multi::Filtration_value& threshold, int value){
	const options_multi::Filtration_value sub_threshold(threshold.begin()+1, threshold.end());
	for (unsigned int i = threshold[0]; i < x.size(); i++){
		add_above(x[i], sub_threshold, value);
	}
}

inline void add_at(grid1d& x, options_multi::value_type at, int value, bool threshold=false){
	x[static_cast<size_t>(at)] += value;
	if (threshold)
		x.back() -= value;
}
inline void add_at(grid2d& x, const options_multi::Filtration_value& at, int value, bool threshold=false){
	x[static_cast<size_t>(at[0])][static_cast<size_t>(at[1])] += value;
	if (threshold){
		x[static_cast<size_t>(at[0])].back() -= value;
		x.back()[static_cast<size_t>(at[1])] -= value;
		x.back().back() += value;
	}
}
inline void add_at(grid3d& x, const options_multi::Filtration_value& at, int value, bool threshold=false){
	x[static_cast<size_t>(at[0])][static_cast<size_t>(at[1])][static_cast<size_t>(at[2])] += value;
	if (threshold){
		x[static_cast<size_t>(at[0])][static_cast<size_t>(at[1])].back() -= value;
		x[static_cast<size_t>(at[0])].back()[static_cast<size_t>(at[2])] -= value;
		x.back()[static_cast<size_t>(at[1])][static_cast<size_t>(at[2])] -= value;

		x[static_cast<size_t>(at[0])].back().back() += value;
		x.back().back()[static_cast<size_t>(at[2])] += value;
		x.back()[static_cast<size_t>(at[1])].back() += value;

		x.back().back().back() -= value;
	}
}
inline void add_at(grid4d& x, const options_multi::Filtration_value& at, int value, bool threshold=false){
	int a =static_cast<int>(at[0]), b =static_cast<int>(at[1]), c =static_cast<int>(at[2]), d =static_cast<int>(at[3]);
	x[a][b][c][d] += value;

	if (threshold){
		x.back()[b][c][d] -= value;
		x[a].back()[c][d] -= value;
		x[a][b].back()[d] -= value;
		x[a][b][c].back() -= value;

		x.back().back()[c][d] += value;
		x.back()[b].back()[d] += value;
		x.back()[b][c].back() += value;

		x[a].back().back()[d] += value;
		x[a].back()[c].back() += value;
		x[a][b].back().back() += value;

		x[a].back().back().back() -= value;
		x.back()[b].back().back() -= value;
		x.back().back()[c].back() -= value;
		x.back().back().back()[d] -= value;

		x.back().back().back().back() += value;

	}
}

template<typename ndarray>
inline void compute_euler(ndarray& out, Simplex_tree_multi& st_multi, bool inverse=false, bool threshold = false){
	for (auto &sh : st_multi.complex_simplex_range()){
		auto filtration = st_multi.filtration((sh));
		int sign = 1-2*(st_multi.dimension(sh) % 2);
		if (inverse){
			add_at(out, filtration, sign, threshold);
		}
		else
			add_above(out, filtration, sign);
	}
}





grid2d get_euler2d(Simplex_tree_multi& st_multi, const std::vector<int> &grid_shape, bool inverse, bool threshold){
	if (grid_shape.size() != 2){
		std::cerr << "Use a 2d grid shape."<<std::endl;
		return grid2d();
	}
	grid2d out = allocate_zero_grid(grid_shape.at(0),grid_shape.at(1));
	compute_euler(out, st_multi, inverse, threshold);
	return out;
}
template<typename ... Args>
grid2d get_euler2d(const intptr_t simplextree_ptr, Args...args){
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	return get_euler2d(st_multi, args...);
}

grid3d get_euler3d(Simplex_tree_multi& st_multi, const std::vector<int> &grid_shape, bool inverse, bool threshold){
	if (grid_shape.size() != 3){
		std::cerr << "Use a 3d grid shape."<<std::endl;
		return grid3d();
	}
	grid3d out = allocate_zero_grid(grid_shape.at(0), grid_shape.at(1), grid_shape.at(2));
	compute_euler(out, st_multi, inverse, threshold);
	return out;
}
template<typename ... Args>
grid3d get_euler3d(const intptr_t simplextree_ptr, Args...args){
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	return get_euler3d(st_multi, args...);
}
grid4d get_euler4d(Simplex_tree_multi& st_multi, const std::vector<int> &grid_shape, bool inverse, bool threshold){
	if (grid_shape.size() != 4){
		std::cerr << "Use a 4d grid shape."<<std::endl;
		return grid4d();
	}
	grid4d out = allocate_zero_grid(grid_shape.at(0), grid_shape.at(1), grid_shape.at(2), grid_shape.at(3));
	compute_euler(out, st_multi, inverse, threshold);
	return out;
}
template<typename ... Args>
grid4d get_euler4d(const intptr_t simplextree_ptr, Args...args){
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	return get_euler4d(st_multi, args...);
}

std::vector<signed_measure> get_signed_measure(
	const intptr_t simplextree_ptr, const std::vector<int> &grid_shape,
	int invariant,const std::vector<int> degrees, bool zero_pad=true
)
{
	auto &st_multi = get_simplextree_from_pointer<interface_multi>(simplextree_ptr);
	int num_parameters = st_multi.get_number_of_parameters();
	if (invariant == 1){ //hilbert
		switch (num_parameters)
		{
		case 2:{
			auto out2 = get_2Dhilbert(st_multi,grid_shape,degrees, 0,1,{},true, zero_pad);
			std::vector<signed_measure> out;
			out.reserve(out2.size());
			for (auto &stuff : out2){
				möbius_inversion(stuff, zero_pad);
				out.push_back(sparsify(stuff));
			}
			return out;
			break;}
		case 3:{
			auto out3 = get_3Dhilbert(st_multi,grid_shape,degrees,0,1,2,{},true, zero_pad);
			std::vector<signed_measure> out;
			out.reserve(out3.size());
			for (auto &stuff : out3){
				möbius_inversion(stuff, zero_pad);
				out.push_back(sparsify(stuff));
			}
			return out;
			break;}
		case 4:{
			auto out4 = get_4Dhilbert(st_multi,grid_shape,degrees,{},true, zero_pad);
			std::vector<signed_measure> out;
			out.reserve(out4.size());
			for (auto &stuff : out4){
				möbius_inversion(stuff, zero_pad);
				out.push_back(sparsify(stuff));
			}
			return out;
			break;}
		default:{
			throw std::invalid_argument("Invalid number of parameters"); 
			break;}
		}
	}
	if (invariant == 2) // euler
	{
		switch (num_parameters)
		{
		case 2:{
			return {sparsify(get_euler2d(st_multi,grid_shape, true, zero_pad))};
			break;}
		case 3:{
			return {sparsify(get_euler3d(st_multi,grid_shape, true, zero_pad))};
			break;}
		case 4:{
			return {sparsify(get_euler4d(st_multi,grid_shape, true, zero_pad))};
			break;}
		default:{
			throw std::invalid_argument("Invalid number of parameters. Has to be <= 4."); 
			break;}
		}	
	}
	if (invariant == 3) // rank invariant
	{
		switch (num_parameters)
		{
		case 2:{
			auto rank = get_2drank_invariant(simplextree_ptr, grid_shape, degrees[0]);
			möbius_inversion(rank, false); // TODO : this is not exact
			return {sparsify(rank)};
			break;}
		
		default:{
			throw std::invalid_argument("Invalid number of parameters, only 2 parameter supported"); 
			break;}
		}
	} 
	throw std::invalid_argument("Invariant not implemented"); 

}

} // namespace rank_invariant