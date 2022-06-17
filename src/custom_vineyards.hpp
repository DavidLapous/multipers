/**
 * @file custom_vineyards.hpp
 * @author Mathieu Carrière, David Loiseaux
 * @brief Python - c++ interface for multipers.py
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef CUSTOM_VINEYARDS_HPP_INCLUDED
#define CUSTOM_VINEYARDS_HPP_INCLUDED

#include "vineyards.h"
#include <iostream>
#include <fstream>
#include <sstream>
// #include "vineyards_trajectories.h"
// #include "approximation.h"
#include "benchmarks.h"
#include <time.h>

typedef vector<pair<int, pair<double, double>>> barcode_eval;
// vineyard_state is a Python-friendly structure that contains all the information of the C++ class VineyardsPersistence
typedef pair<pair<pair<barcode, barcode_eval>, vector<int>>, pair<pair<pair<boundary_matrix, pair<vector<int>, vector<int>>>, pair<boundary_matrix, pair<vector<int>, vector<int>>>>, pair<vector<int>, vector<int>>>> vineyard_state;

vineyard_state lower_star_vineyards_update(boundary_matrix structure, vector<double> filter, vector<int> dimensions, // dimensions is associated to structure if compute_barcode is true else to R
																					 bool compute_barcode = false,
																					 boundary_matrix R = {}, boundary_matrix U = {}, vector<int> permutation = {}, barcode bc = {}, vector<int> bc_inv = {}, vector<int> row_map_R = {}, vector<int> row_map_inv_R = {}, vector<int> row_map_U = {}, vector<int> row_map_inv_U = {})
{

	// TODO: check if VPP is really necessary
	VineyardsPersistence VP;

	if (compute_barcode)
	{
		VineyardsPersistence VPP(structure, dimensions, filter, true, false);
		VPP.initialize_barcode();
		VP = VPP;
	}
	else
	{
		std::cout << std::endl;
		std::cout << "New update: " << std::endl;
		VineyardsPersistence VPP(structure, R, U, dimensions, permutation, bc, bc_inv, row_map_R, row_map_inv_R, row_map_U, row_map_inv_U, true);
		VPP.update(filter);
		VP = VPP;
	}

	boundary_matrix RR = VP.P.first.matrix.matrix;
	vector<int> rowmR = VP.P.first.row_map;
	vector<int> rowmRi = VP.P.first.row_map_inv;
	pair<vector<int>, vector<int>> PRW(rowmR, rowmRi);
	pair<boundary_matrix, pair<vector<int>, vector<int>>> PRR(RR, PRW);

	boundary_matrix UU = VP.P.second.matrix.matrix;
	vector<int> rowmU = VP.P.second.row_map;
	vector<int> rowmUi = VP.P.second.row_map_inv;
	pair<vector<int>, vector<int>> PUW(rowmU, rowmUi);
	pair<boundary_matrix, pair<vector<int>, vector<int>>> PUU(UU, PUW);

	vector<int> new_dims = VP.P.first.dimensions;
	vector<int> new_perm = VP.permutation;
	barcode new_bc = VP.P.first.bc;
	vector<int> new_bc_inv = VP.P.first.bc_inv;
	VP.get_diagram();
	pair<vector<int>, vector<int>> PV(new_dims, new_perm);
	pair<pair<boundary_matrix, pair<vector<int>, vector<int>>>, pair<boundary_matrix, pair<vector<int>, vector<int>>>> PB(PRR, PUU);
	pair<barcode, barcode_eval> PBC(new_bc, VP.dgm);
	pair<pair<pair<boundary_matrix, pair<vector<int>, vector<int>>>, pair<boundary_matrix, pair<vector<int>, vector<int>>>>, pair<vector<int>, vector<int>>> PC(PB, PV);
	pair<pair<barcode, barcode_eval>, vector<int>> PD(PBC, new_bc_inv);
	vineyard_state VS(PD, PC);

	return VS;
}

vector<vector<vector<double>>> vineyards(const vector<vector<double>> &vertices_values, const string &complex_fn, const int &discard_inf)
{
	cout << "Initializing variables...";
	clock_t elapsed = clock();
	int max_dim = 0;
	string line;
	ifstream in_complex_file(complex_fn);
	boundary_matrix structure;
	vector<int> dims;
	map<vector<int>, int> positions;
	int ID = 0;
	cout << " Done !" << endl;

	cout << "Reading file...";
	while (getline(in_complex_file, line))
	{
		stringstream s(line);
		int vertex;
		boundary bd, matrix_bd;
		while (s >> vertex)
			bd.push_back(vertex);
		if (bd.size() == 1)
		{
			structure.push_back(matrix_bd);
			dims.push_back(bd.size() - 1);
		}
		else
		{
			for (uint i = 0; i < bd.size(); i++)
			{
				vector<int> new_bd = bd;
				new_bd.erase(new_bd.begin() + i);
				matrix_bd.push_back(positions[new_bd]);
			}
			structure.push_back(matrix_bd);
			int bds = bd.size() - 1;
			dims.push_back(bds);
			max_dim = max(max_dim, bds);
		}
		positions.insert(pair<vector<int>, int>(bd, ID));
		ID += 1;
	}
	cout << " Done!" << endl;

	if (discard_inf == 0)
		max_dim += 1;
	VineyardsPersistence VP;
	bool first = true;
	int time = 0;

	vector<vector<pair<pair<int, int>, pair<double, double>>>> vineyards;

	for (uint l = 0; l < vertices_values.size(); l++)
	{

		vector<double> filt = vertices_values[l];
		if (first)
		{
			VineyardsPersistence VPP(structure, dims, filt, true, false);
			VPP.initialize_barcode();
			vector<pair<pair<int, int>, pair<double, double>>> tmp;
			for (uint i = 0; i < VPP.P.first.bc.size(); i++)
				vineyards.push_back(tmp);
			VP = VPP;
			first = false;
		}
		else
		{
			cout << "Updating barcode, line " << l << " over "<< vertices_values.size() <<"..." << flush;
			VP.update(filt);
			cout << "\r";
		}

		VP.get_diagram();
		for (int d = 0; d < max_dim; d++)
		{
			for (uint i = 0; i < VP.dgm.size(); i++)
			{
				if (VP.dgm[i].first == d)
				{
					pair<double, double> pt(VP.dgm[i].second.first, VP.dgm[i].second.second);
					vineyards[i].push_back(pair<pair<int, int>, pair<double, double>>(pair<int, int>(time, d), pt));
				}
			}
		}
		time += 1;
	}
	cout  <<"Updating barcode, line " << vertices_values.size() << " over "<< vertices_values.size() <<"..." <<  "Done !" << endl << "Generating output...";
	vector<vector<vector<double>>> vineout(max_dim);

	for (int d = 0; d < max_dim; d++)
	{
		for (uint i = 0; i < vineyards.size(); i++)
		{
			vector<pair<pair<int, int>, pair<double, double>>> vine = vineyards[i];
			int Lv = vine.size();
			bool trivial = true;
			for (int j = 0; j < Lv; j++)
			{
				if (vine[j].second.second != vine[j].second.first)
				{
					trivial = false;
					break;
				}
			}
			if (!trivial)
			{
				vector<double> traj;
				for (int j = 0; j < Lv; j++)
				{
					if (vine[j].first.second == d)
					{
						if (vine[j].second.second >= DBL_MAX)
						{
							if (discard_inf == 0)
							{
								traj.push_back(vine[j].second.first);
								traj.push_back(10000000000);
								traj.push_back(vine[j].first.first);
							}
						}
						else
						{
							traj.push_back(vine[j].second.first);
							traj.push_back(vine[j].second.second);
							traj.push_back(vine[j].first.first);
						}
					}
				}
				vineout[d].push_back(traj);
			}
		}
	}
	elapsed = clock() - elapsed;
	cout << " Done! It took "<< ((float)elapsed)/CLOCKS_PER_SEC << " seconds." << endl;

	return vineout;
};

#endif // CUSTOM_VINEYARDS_HPP_INCLUDED
