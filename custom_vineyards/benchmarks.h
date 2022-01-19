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
#include "approximation.h"


double time_vineyard_alt(boundary_matrix &B, const vector<vector<double>> &filters_list, double precision, pair<vector<double>, vector<double>> &box, bool threshold = false, bool multithread = false){

	auto elapsed = clock();
	vineyard_alt(B, filters_list,precision, box,threshold, multithread);
	elapsed = clock() - elapsed;
	return ((float)elapsed)/CLOCKS_PER_SEC;
}



double time_approximation(boundary_matrix &B, const vector<vector<double>> &filters_list, const double precision, const pair<vector<double>, vector<double>> &box, const bool threshold = false, const bool keep_order = false, const  bool complete=true, const bool multithread = false, const bool verbose = false){
	if(verbose)	cout << "Starting approx..." << flush;
	auto elapsed = clock();
	approximation_vineyards(B,filters_list,precision,box,threshold,keep_order, complete,multithread, false);
	elapsed = clock() - elapsed;
	auto time = ((double)elapsed) / CLOCKS_PER_SEC;
	if(verbose)	cout << " Done ! It took " <<  time << "seconds." << endl;
	return time;
}
#endif // BENCHMARKS_H_INCLUDED
