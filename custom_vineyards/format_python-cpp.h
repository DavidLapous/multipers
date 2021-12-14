/**
 * @file format_python-cpp.h
 * @author David Loiseaux
 * @brief Functions that change the format of data to communicate between C++ and python.
 * 
 * @copyright Copyright (c) 2021 Inria
 * 
 */

#ifndef FORMAT_PYTHON_CPP_H_INCLUDED
#define FORMAT_PYTHON_CPP_H_INCLUDED
#include "dependences.h"
#include "combinatory.h"

void get_dimensions_from_structure(boundary_matrix &B, vector<int> &dimensions){
// 	if(sparse){
// 		uint n = dimensions.size();
// 		boundary_matrix sparse(n, vector<int>(n));
// 		for(uint i = 0; i<n;i++){
// 			uint dim=0;
// 			uint j=i;
// 			while(!B[j].empty()){
// 				j = B[j][0];
// 				dim++;
// 			}
// 			dimensions[i] = dim;
// 		}
// 		return;
// 	}
	for(uint simplex=0;simplex<dimensions.size();simplex++)
		dimensions[simplex] = max((int)(B[simplex].size())-1,0);
	return;
}

void get_dimensions_from_structure(vector<vector<vector<int>>> &B, vector<uint> &dimensions){
	for(uint simplex=0;simplex<dimensions.size();simplex++)
		dimensions[simplex] = max((int)(B[simplex].size())-1,0);
	return;
}





void order_boundary_matrix_by_dimension(boundary_matrix &B, vector<int> &dimensions){
	get_dimensions_from_structure(B, dimensions);
	vector<int> p = sort_to_permutation(dimensions);
	auto list_transpositions =  permutation_to_transpositions(p);
	for (const auto &transposition : list_transpositions){
		B[transposition.first].swap(B[transposition.second]);
	}
	return;
}

string simplex_to_string(const vector<vector<int>> &v){
	stringstream ss;
	for(size_t i = 0; i < v.size(); ++i){
		for(size_t j=0; j< v[i].size();j++){
			if(j!=0) ss<< " ";
			ss << v[i][j];
		}
		if(i != 0) ss << ",";
	}
	return ss.str();
}





// Lexical order + dimension
bool simplex_comp(const vector<uint> &s1, const vector<uint> &s2){
	if (s1.size() < s2.size()) return true;
	if (s1.size() > s2.size()) return false;
	for (uint i=0; i< s1.size(); i++){
		if (s1[i] < s2[i]) return true;
		if (s1[i] > s2[i]) return false;
	}
	return false;
}

// Converts a simplex into an uint for dictionary
uint simplex_to_uint(vector<uint> &simplex, uint scale){
	sort(simplex.begin(), simplex.end());
	uint output = 0;
	for(uint i=0; i<simplex.size(); i++){
		output += simplex[i]*pow(scale,i);
	}
	return output;
}

// converts the simplex j in boundary of simplex to an uint for dictionnary
uint simplex_to_uint(vector<uint> &simplex, int j, uint scale){
	sort(simplex.begin(), simplex.end());
	uint output = 0;
	bool passed_through_j=0;
	for(int i=0; i<(int)simplex.size(); i++){
		if(i==j){
			passed_through_j = 1;
			continue;
		}
		output += simplex[i]*pow(scale,i - passed_through_j);
	}
	return output;
}


vector<vector<uint>> list_simplicies_to_sparse_boundary_matrix(vector<vector<uint>> &list_simplices){
	uint num_simplices = list_simplices.size();
	uint scale = pow(10,ceil(log10(num_simplices)));
	for(uint i=0; i<num_simplices; i++){
		sort(list_simplices[i].begin(), list_simplices[i].end());
	}
	stable_sort(list_simplices.begin(), list_simplices.end(), simplex_comp);
	vector<vector<uint>> output(num_simplices);

	// Dictionary to store simplex ids. simplex [0,2,4] number is simplex_id[024]; that's why we needed to sort first
	unordered_map<uint, uint> simplex_id;
	for (uint i = 0; i<num_simplices; i++){
// 		populate the dictionary with this simplex
		simplex_id.emplace(simplex_to_uint(list_simplices[i], scale), i);
		// If simplex is of dimension 0, there is no boundary
		if(list_simplices[i].size() <=1) continue;
// 		Fills the output matrix with the boundary of simplex cursor
		for(uint j = 0; j<list_simplices[i].size(); j++){
			uint child_id = simplex_id[simplex_to_uint(list_simplices[i], j, scale)];
			output[i].push_back(child_id);
		}
	}
	for(uint i=0; i<num_simplices; i++){
		sort(output[i].begin(), output[i].end());
	}
	stable_sort(output.begin(), output.end(), simplex_comp);
	return output;
}





pair<vector<vector<uint>>, vector<vector<double>>> list_simplices_ls_filtration_to_sparse_boundary_filtration(vector<vector<uint>> &list_simplices, vector<vector<double>> &to_complete_filtration, vector<uint> &filtration_to_order){
	uint num_simplices = list_simplices.size();
	uint scale = pow(10,ceil(log10(num_simplices))); // for dictionary hashmap
	uint filtration_dimension = to_complete_filtration.size();

	for(uint i=0; i<num_simplices; i++){
		sort(list_simplices[i].begin(), list_simplices[i].end());
	}
// 	disp_vect(list_simplices);
// 	stable_sort(list_simplices.begin(), list_simplices.end(), simplex_comp);

	//sort list_simplices with filtration
	vector<uint> p = sort_to_permutation<vector<uint>>(list_simplices, &simplex_comp); // of size num_simplices

	vector<vector<uint>> boundary(num_simplices);
	// WARNING We assume here that point filtration has the same order as the ordered list of simplices.
// 	This fills the filtration of the 0-skeleton by points_filtration
	vector<vector<double>> ls_filters(filtration_dimension, vector<double>(num_simplices, -DBL_MAX));
	for (uint i=0; i<filtration_dimension; i++)
		for(uint j =0; j<to_complete_filtration[i].size();j++)
			ls_filters[i][j] = to_complete_filtration[i][j];

	for(const uint index : filtration_to_order){
		compose(ls_filters[index],p); // permute filters the same as simplices
	}
// 	disp_vect(p);
// 	disp_vect(list_simplices);
	// Dictionary to store simplex ids. simplex [0,2,4] number is simplex_id[024]; that's why we needed to sort first
	unordered_map<uint, uint> simplex_id;
	for (uint i = 0; i<num_simplices; i++){
// 		populate the dictionary with this simplex
		simplex_id.emplace(simplex_to_uint(list_simplices[i], scale), i); // stores the id of the simplex

		// If simplex is of dimension 0, there is no boundary
		if(list_simplices[i].size() <=1) continue;
// 		Fills the output matrix with the boundary of simplex cursor, and computes filtration of the simplex
		for(uint j = 0; j<list_simplices[i].size(); j++){
			uint child_id = simplex_id[simplex_to_uint(list_simplices[i], j, scale)]; // computes the id of the child
			boundary[i].push_back(child_id); // add this child to the boundary
			for(uint k=0; k<filtration_dimension; k++)
				ls_filters[k][i] = max(ls_filters[k][i], ls_filters[k][child_id]); // this simplex filtration is greater than the childs filtration in the ls case
		}
	}
	for(uint i=0; i<num_simplices; i++){
		sort(boundary[i].begin(), boundary[i].end());
	}
// 	stable_sort(boundary.begin(), boundary.end(), simplex_comp);
	return {boundary, ls_filters};
}




#endif // FORMAT_PYTHON_CPP_H_INCLUDED
