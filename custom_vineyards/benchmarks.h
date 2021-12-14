/**
 * @file benchmarks.h
 * @author David Loiseaux	
 * @brief Functions to benchmark specific part of functions from vineyard_trajectories.h
 * 
 * @copyright Copyright (c) 2021 Inria
 * 
 */
#ifndef BENCHMARKS_H_INCLUDED
#define BENCHMARKS_H_INCLUDED

#include "dependences.h"
#include "vineyards_trajectories.h"


double time_vineyard_alt(boundary_matrix &B, const vector<vector<double>> &filters_list, double precision, pair<vector<double>, vector<double>> &box, bool threshold = false, vector<uint> nlines={}, bool multithread = false){
	const uint filtration_dimension = filters_list.size();
	uint number_simplices = B.size();
	vector<int> simplicies_dimensions(number_simplices);
	get_dimensions_from_structure(B, simplicies_dimensions);
	bool lower_star = false;
	if(filters_list[0].size() < number_simplices ) lower_star = true;

	vector<double> filter(number_simplices); // container of filters

	vector<uint> size_line(filtration_dimension-1);
	if (nlines.size() < filtration_dimension-1)
		for(uint i=0;i<filtration_dimension-1;i++)
			size_line[i] = (uint)(ceil((abs( (box.second[i] - box.first.back()) - (box.first[i] - box.second.back()) ) / precision)));
	else{
		for(uint i=0;i<filtration_dimension-1;i++)
			size_line[i] = nlines[i];
		// cout << "Custom size" << endl;
	}
// 	disp_vect(size_line);
	uint number_of_line = prod(size_line);
	auto &basepoint = box.first;
	for(uint i=0; i<basepoint.size()-1; i++) basepoint[i] -= box.second.back();
	basepoint.back() = 0;

	line_to_filter(basepoint, filters_list, filter, true);
	vector<uint> position(filtration_dimension-1, 0); // where is the cursor in the output matrix

	VineyardsPersistence persistence(B, simplicies_dimensions, filter, lower_star, false, false);
	persistence.initialize_barcode(false, false);
	persistence.get_diagram();

	auto &first_barcode = persistence.dgm;
	uint max_dimension = first_barcode.back().first; // filtered by dimension so last one is of maximal dimension
	uint number_of_features = first_barcode.size();
	vector<vector<vector<interval>>> output(max_dimension+1);

	vector<uint> number_of_feature_of_dimension(max_dimension+1);
	for(uint i=0; i< number_of_features; i++){
		number_of_feature_of_dimension[first_barcode[i].first]++;
	}
	for(uint i=0; i<max_dimension+1;i++){
		output[i] = vector<vector<interval>>(number_of_feature_of_dimension[i], vector<interval>(number_of_line));
	}

// 	cout << "Number of lines : " << number_of_line << endl;
// 	cout << "Number of simplices : " << number_simplices << endl;
	auto elapsed = clock();
	vineyard_alt_recursive(output, persistence, basepoint, position, 0, filter, filters_list,precision,box, size_line, false, threshold, multithread);
	elapsed = clock() - elapsed;
	return ((float)elapsed)/CLOCKS_PER_SEC;
}
#endif // BENCHMARKS_H_INCLUDED
