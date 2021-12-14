/**
 * @file vineyards.h
 * @author Mathieu Carrière, David Loiseaux
 * @brief Core vineyards functions
 * 
 * @copyright Copyright (c) 2021 Inria
 * 
 */
#ifndef VINEYARDS_H_INCLUDED
#define VINEYARDS_H_INCLUDED


#include "combinatory.h"
#include "debug.h"
#include "dependences.h"

using namespace std;

typedef vector<int> boundary;
typedef vector<boundary> boundary_matrix;
typedef vector<pair<int, pair<int, int>>> barcode;


/**
 * @brief Sparse boundary matrix, with fast column swapping
 * 
 */
class FastSwapSparseBoundaryMatrix
{

public:
	FastSwapSparseBoundaryMatrix()
	{
		// boundary_matrix tmp;
		// matrix = tmp;
	}
	FastSwapSparseBoundaryMatrix(int n)
	{
		// boundary_matrix tmp(n);
		// matrix = tmp;
		matrix.resize(n);
	}
	FastSwapSparseBoundaryMatrix(boundary_matrix input) { matrix = input; }
	void swapCols(int i, int j)
	{
		// boundary tmp(matrix[i]);
		// matrix[i] = matrix[j];
		// matrix[j] = tmp;
		matrix[i].swap(matrix[j]);
	}
	int num_simplices() { return matrix.size(); }
	boundary return_boundary(int i) { return matrix[i]; }
	void update_boundary(int i, boundary L) { 
		// matrix[i] = L; 
		matrix[i].swap(L);
	};
	void set_to_false(int i, int j) { matrix[i].erase(matrix[i].begin() + j); }

	boundary_matrix matrix;
};

/**
 * @brief Sparse boundary matrix with fast rows and column swapping.
 * 
 */
class SparseBoundaryMatrix
{

public:
	SparseBoundaryMatrix()
	{
		FastSwapSparseBoundaryMatrix matrix_;
		this->matrix = matrix_;
		vector<int> tmp;
		this->row_map = tmp;
		this->row_map_inv = tmp;
		this->dimensions = tmp;
		this->bc_inv = tmp;
		barcode tmpb;
		this->bc = tmpb;
	}
	SparseBoundaryMatrix(boundary_matrix input)
	{
		FastSwapSparseBoundaryMatrix matrix_(input);
		this->matrix = matrix_;
		int n = matrix_.num_simplices();
		vector<int> tmp(n);
		for (int i = 0; i < n; i++)
		{
			tmp[i] = i;
		}
		this->row_map = tmp;
		this->row_map_inv = tmp;
		this->dimensions = tmp;
		this->bc_inv = tmp;
		barcode tmpb;
		this->bc = tmpb;
	}
	SparseBoundaryMatrix(int n)
	{
		FastSwapSparseBoundaryMatrix matrix_(n);
		this->matrix = matrix_;
		vector<int> tmp(n);
		for (int i = 0; i < n; i++)
		{
			tmp[i] = i;
		}
		this->row_map = tmp;
		this->row_map_inv = tmp;
		this->dimensions = tmp;
		this->bc_inv = tmp;
		barcode tmpb;
		this->bc = tmpb;
	}
	SparseBoundaryMatrix(const SparseBoundaryMatrix &input)
	{
		this->matrix = input.matrix;
		this->row_map = input.row_map;
		this->row_map_inv = input.row_map_inv;
		this->bc_inv = input.bc_inv;
		this->bc = input.bc;
		this->dimensions = input.dimensions;
	}

	// Erase entry in constant time
	/**
	 * @brief Set the to false object
	 * 
	 * @param i 
	 * @param k 
	 */
	void set_to_false(int i, int k) { this->matrix.set_to_false(i, k); }

	/**
	 * @brief Swap columns @p col1 and @p col2 in constant time
	 * 
	 * @param col1 
	 * @param col2 
	 */
	void swapCols(int col1, int col2)
	{
		this->matrix.swapCols(col1, col2);
		swap(this->dimensions[col1], this->dimensions[col2]);
	}

	/**
	 * @brief Swap rows @p row1 and @p row2 in constant time
	 * 
	 * @param row1 
	 * @param row2 
	 */
	void swapRows(int row1, int row2)
	{
		swap(this->row_map_inv[row1], this->row_map_inv[row2]);
		swap(this->row_map[this->row_map_inv[row2]], this->row_map[this->row_map_inv[row1]]);
	}

	// Replace column in constant time
	/**
	 * @brief Replace column @p i in constant time by @p L
	 * 
	 * @param i 
	 * @param L 
	 */
	void update_boundary(int i, boundary &L) { this->matrix.update_boundary(i, L); }

	// Add columns in linear time
	void sparse_add_in_Z2(int i, int j)
	{
		boundary L1 = this->matrix.return_boundary(i);
		boundary L2 = this->matrix.return_boundary(j);
		int u = 0;
		int v = 0;
		int l1 = L1.size();
		int l2 = L2.size();
		boundary T;
		while (u < l1 and v < l2)
		{
			if (L1[u] < L2[v])
			{
				T.push_back(L1[u]);
				u += 1;
			}
			else if (L2[v] < L1[u])
			{
				T.push_back(L2[v]);
				v += 1;
			}
			else
			{
				u += 1;
				v += 1;
			}
		}
		if (u == l1)
		{
			while (v < l2)
			{
				T.push_back(L2[v]);
				v += 1;
			}
		}
		if (v == l2)
		{
			while (u < l1)
			{
				T.push_back(L1[u]);
				u += 1;
			}
		}
		this->matrix.update_boundary(j, T);
	}

	// 
	/**
	 * @brief Access entry [ @p i, @p j ]  in linear time
	 * 
	 * @param i 
	 * @param j 
	 * @return pair<bool, int> 
	 */
	pair<bool, int> entry(int i, int j) 
	{
		boundary L = this->matrix.return_boundary(i);
		for (unsigned int k = 0; k < L.size(); k++)
		{
			if (this->row_map[L[k]] == j)
			{
				pair<bool, int> P(true, k);
				return P;
			}
		}
		pair<bool, int> P(false, -1);
		return P;
	}
	/**
	 * @brief Gets the number of simplices.
	 * 
	 * @return Uint 
	 */
	uint num_simplices() { return this->matrix.num_simplices(); }
	boundary return_boundary(int i) { return this->matrix.return_boundary(i); }
	void display_map()
	{
		for (unsigned int i = 0; i < this->row_map.size(); i++)
			cout << this->row_map[i] << " ";
		cout << endl;
	}
	void display_bc()
	{
		for (unsigned int i = 0; i < this->bc.size(); i++)
			cout << " " << bc[i].first << " " << bc[i].second.first << " " << bc[i].second.second << endl;
	}
	void display_bc_inv()
	{
		for (unsigned int i = 0; i < this->bc_inv.size(); i++)
			cout << this->bc_inv[i] << " ";
		cout << endl;
	}
	void display()
	{
		for (int i = 0; i < this->matrix.num_simplices(); i++)
		{
			boundary tmp = this->matrix.return_boundary(i);
			for (unsigned int j = 0; j < tmp.size(); j++)
				cout << this->row_map[tmp[j]] << " ";
			cout << endl;
		}
	}

	FastSwapSparseBoundaryMatrix matrix;

	/**
	 * @brief row_map is the inverse of row_map_inv; it contains the row index of the simplices; 
	 * row_map_inv contains the IDs of the simplices corresponding to each row, ex: [ID_sigma_1, ID_sigma_2, ..., ID_sigma_N]
	 */
	vector<int> row_map, row_map_inv;
	
	barcode bc;	///<  contains the pairings after a persistence computation
	vector<int> bc_inv; ///<  inverse of bc

	vector<int> dimensions;	///<  contains the simplex dimensions of the columns
};


typedef pair<SparseBoundaryMatrix, SparseBoundaryMatrix> RU_decomposition;


/**
 * @brief Returns the sum in \f$ \mathbb{F}_2\f$ of @p L1 and @p L2 . Sparse matrix format.
 * 
 * @param L1 
 * @param L2 
 * @return boundary 
 */
boundary sparse_add_in_Z2(boundary L1, boundary L2)
{
	int u = 0;
	int v = 0;
	int l1 = L1.size();
	int l2 = L2.size();
	boundary T;
	T.clear();
	while (u < l1 and v < l2)
	{
		if (L1[u] < L2[v])
		{
			T.push_back(L1[u]);
			u += 1;
		}
		else if (L2[v] < L1[u])
		{
			T.push_back(L2[v]);
			v += 1;
		}
		else
		{
			u += 1;
			v += 1;
		}
	}
	if (u == l1)
	{
		while (v < l2)
		{
			T.push_back(L2[v]);
			v += 1;
		}
	}
	if (v == l2)
	{
		while (u < l1)
		{
			T.push_back(L1[u]);
			u += 1;
		}
	}
	return T;
};

// compute_barcode_sparse fills the barcode bc of B by reference and outputs the R,U decomposition
// you have to specify whether B already had swapped rows or not 
// TODO : get a better algo
/**
 * @brief Computes RU decomposition of the first line.
 * 
 * @param B 		Sparse boundary matrix of the chain complex
 * @param swapped 	Is the matrix ordered
 * @param verbose 	
 * @return pair<SparseBoundaryMatrix, SparseBoundaryMatrix> 
 */
pair<SparseBoundaryMatrix, SparseBoundaryMatrix> compute_barcode_sparse(SparseBoundaryMatrix &B, bool swapped = false, bool verbose = false)
{

	int n = B.num_simplices();
	boundary_matrix R(n);
	vector<int> V(n);
	B.bc.clear();

	SparseBoundaryMatrix RR(n);
	SparseBoundaryMatrix UU(n);

	for (int i = 0; i < n; i++)
	{
		boundary l = {i};
		UU.update_boundary(i, l);
	} // Initialize UU as the identity

	boundary empty;
	empty.clear();
	int bcidx = 0;
	boundary_matrix BB;

	// if B has been row-swapped, construct another BB which is the ordered version of B ---> B and BB have same persistence, but lowest index can be accessed in constant time with BB
	// TODO: it should be possible to directly swap the rows of B and maintain B.row_map and B.row_map_inv so that B becomes ordered and BB does not have to be created
	if (swapped)
	{
		for (int i = 0; i < n; i++)
		{
			boundary L = B.return_boundary(i);
			int Ls = L.size();
			for (int l = 0; l < Ls; l++)
			{
				L[l] = B.row_map[L[l]];
			}
			sort(L.begin(), L.end());
			BB.push_back(L);
		}
	}

	if (verbose)
	{
		cout << endl
			 << "Input matrix :" << endl;
		for (uint i = 0; i < BB.size(); i++)
		{
			for (uint j = 0; j < BB[i].size(); j++)
				cout << BB[i][j] << " ";
			cout << endl;
		}
	}

	// Standard persistence algorithm, cf. Edelsbrunner & Harer VII.2

	for (int i = 0; i < n; i++)
	{

		V[i] = i;
		boundary L;
		if (!swapped)
			L = B.return_boundary(i);
		else
			L = BB[i];
		int j = i;
		boundary LL = empty;

		if (L.size() > 0)
		{
			j = L.back();
			LL = R[j];
		}

		while (L.size() > 0 && LL.size() > 0)
		{
			L = sparse_add_in_Z2(LL, L);
			UU.sparse_add_in_Z2(i, V[j]);
			if (L.size() > 0)
			{
				j = L.back();
				LL = R[j];
			}
		}

		if (L.size() > 0 && j < n)
		{
			V[j] = i;
			V[i] = -1;
			R[j] = L;
			B.bc[B.bc_inv[j]].second.second = i;
			B.bc_inv[i] = B.bc_inv[j];
		}
		else
		{
			pair<int, int> pt(i, -1);
			B.bc.push_back(pair<int, pair<int, int>>(B.dimensions[i], pt));
			B.bc_inv[i] = bcidx;
			bcidx += 1;
		}
	}

	for (int i = 0; i < n; i++)
	{
		boundary L = R[i];
		if (V[i] != -1 && L.size() > 0)
		{
			int j = V[i];
			RR.update_boundary(j, L);
		}
	}

	RR.bc = B.bc;
	RR.bc_inv = B.bc_inv;
	RR.dimensions = B.dimensions;

	//   pair<SparseBoundaryMatrix,SparseBoundaryMatrix> PP(RR,UU);

	if (verbose)
	{
		cout << "Output matrix :" << endl;
		RR.display();
	}

	return {RR, UU};
};
/**
 * @brief Vineyard update of a permutation ( @p i, @p i + 1 ) of the matrix @p P.
 * 
 * @param i 
 * @param P 
 * @param verbose 
 */
void vineyard_update(int i, pair<SparseBoundaryMatrix, SparseBoundaryMatrix> &P, bool verbose = false)
{

	pair<bool, int> u(P.second.entry(i, i + 1));
	SparseBoundaryMatrix &RR = P.first;
	SparseBoundaryMatrix &UU = P.second;

	int ineg = RR.return_boundary(i).size();
	int iineg = RR.return_boundary(i + 1).size();
	int bci = RR.bc_inv[i];
	int bcii = RR.bc_inv[i + 1];
	int n = RR.num_simplices();
	vector<int> matching(n);
	for (int i = 0; i < n; i++)
		matching[i] = i;

	if (RR.dimensions[i] == RR.dimensions[i + 1])
	{

		if (ineg == 0 && iineg == 0)
		{

			int k = RR.bc[RR.bc_inv[i]].second.second;
			int l = RR.bc[RR.bc_inv[i + 1]].second.second;
			pair<bool, int> r(false, 0);
			if (l != -1)
				r = RR.entry(l, i);
			//	   pair<bool, int> u = U.entry(i, i + 1);

			if (u.first)
				UU.set_to_false(i, u.second);

			if (k != -1 && l != -1 && r.first)
			{ // Case 1.1
				if (k < l)
				{ // Case 1.1.1.
					if (verbose)
						cout << "Case 1.1.1" << endl;
					RR.swapCols(i, i + 1);
					RR.swapRows(i, i + 1);
					RR.sparse_add_in_Z2(k, l);
					UU.swapRows(i, i + 1);
					UU.swapCols(i, i + 1);
					UU.sparse_add_in_Z2(l, k);

					matching[i] = i + 1;
					matching[i + 1] = i;
					double di = RR.bc[bci].second.second;
					double dii = RR.bc[bcii].second.second;
					double dmi, dmii;
					if (di == -1)
						dmi = -1;
					else
						dmi = matching[di];
					if (dii == -1)
						dmii = -1;
					else
						dmii = matching[dii];
					pair<int, int> pti(matching[RR.bc[bci].second.first], dmi);
					RR.bc[bci].second = pti;
					pair<int, int> ptii(matching[RR.bc[bcii].second.first], dmii);
					RR.bc[bcii].second = ptii;
					swap(RR.bc_inv[i], RR.bc_inv[i + 1]);
					return;
				}
				else if (l < k)
				{ // Case 1.1.2
					if (verbose)
						cout << "Case 1.1.2" << endl;
					RR.swapCols(i, i + 1);
					RR.swapRows(i, i + 1);
					RR.sparse_add_in_Z2(l, k);
					UU.swapRows(i, i + 1);
					UU.swapCols(i, i + 1);
					UU.sparse_add_in_Z2(k, l);
					return;
				}
			}
			// Case 1.2
			if (verbose)
				cout << "Case 1.2" << endl;
			RR.swapCols(i, i + 1);
			RR.swapRows(i, i + 1);
			UU.swapRows(i, i + 1);
			UU.swapCols(i, i + 1);

			if (!(k == -1 && l != -1 && r.first))
			{
				matching[i] = i + 1;
				matching[i + 1] = i;
				double di = RR.bc[bci].second.second;
				double dii = RR.bc[bcii].second.second;
				double dmi, dmii;
				if (di == -1)
					dmi = -1;
				else
					dmi = matching[di];
				if (dii == -1)
					dmii = -1;
				else
					dmii = matching[dii];
				pair<int, int> pti(matching[RR.bc[bci].second.first], dmi);
				RR.bc[bci].second = pti;
				pair<int, int> ptii(matching[RR.bc[bcii].second.first], dmii);
				RR.bc[bcii].second = ptii;
				swap(RR.bc_inv[i], RR.bc_inv[i + 1]);
				return;
			}
		}

		else if (ineg > 0 && iineg > 0)
		{

			//	   pair<bool, int> u = U.entry(i, i + 1);
			int lowi = P.first.bc[P.first.bc_inv[i]].first;
			int lowii = P.first.bc[P.first.bc_inv[i + 1]].first;

			if (u.first)
			{ // Case 2.1
				if (lowi < lowii)
				{ // Case 2.1.1
					if (verbose)
						cout << "Case 2.1.1" << endl;
					RR.sparse_add_in_Z2(i, i + 1);
					RR.swapCols(i, i + 1);
					RR.swapRows(i, i + 1);
					UU.sparse_add_in_Z2(i + 1, i);
					UU.swapRows(i, i + 1);
					UU.swapCols(i, i + 1);

					matching[i] = i + 1;
					matching[i + 1] = i;
					double di = RR.bc[bci].second.second;
					double dii = RR.bc[bcii].second.second;
					double dmi, dmii;
					if (di == -1)
						dmi = -1;
					else
						dmi = matching[di];
					if (dii == -1)
						dmii = -1;
					else
						dmii = matching[dii];
					pair<int, int> pti(matching[RR.bc[bci].second.first], dmi);
					RR.bc[bci].second = pti;
					pair<int, int> ptii(matching[RR.bc[bcii].second.first], dmii);
					RR.bc[bcii].second = ptii;
					swap(RR.bc_inv[i], RR.bc_inv[i + 1]);
					return;
				}
				else
				{ // Case 2.1.2
					if (verbose)
						cout << "Case 2.1.2" << endl;
					RR.sparse_add_in_Z2(i, i + 1);
					RR.swapCols(i, i + 1);
					RR.swapRows(i, i + 1);
					RR.sparse_add_in_Z2(i, i + 1);
					UU.sparse_add_in_Z2(i + 1, i);
					UU.swapRows(i, i + 1);
					UU.swapCols(i, i + 1);
					UU.sparse_add_in_Z2(i + 1, i);
					return;
				}
			}
			else
			{ // Case 2.2
				if (verbose)
					cout << "Case 2.2" << endl;
				RR.swapCols(i, i + 1);
				RR.swapRows(i, i + 1);
				UU.swapRows(i, i + 1);
				UU.swapCols(i, i + 1);

				matching[i] = i + 1;
				matching[i + 1] = i;
				double di = RR.bc[bci].second.second;
				double dii = RR.bc[bcii].second.second;
				double dmi, dmii;
				if (di == -1)
					dmi = -1;
				else
					dmi = matching[di];
				if (dii == -1)
					dmii = -1;
				else
					dmii = matching[dii];
				pair<int, int> pti(matching[RR.bc[bci].second.first], dmi);
				RR.bc[bci].second = pti;
				pair<int, int> ptii(matching[RR.bc[bcii].second.first], dmii);
				RR.bc[bcii].second = ptii;
				swap(RR.bc_inv[i], RR.bc_inv[i + 1]);

				return;
			}
		}

		else if (ineg > 0 && iineg == 0)
		{
			//	   pair<bool, int> u = U.entry(i, i + 1);
			if (u.first)
			{ // Case 3.1
				if (verbose)
					cout << "Case 3.1" << endl;
				RR.sparse_add_in_Z2(i, i + 1);
				RR.swapCols(i, i + 1);
				RR.swapRows(i, i + 1);
				RR.sparse_add_in_Z2(i, i + 1);
				UU.sparse_add_in_Z2(i + 1, i);
				UU.swapRows(i, i + 1);
				UU.swapCols(i, i + 1);
				UU.sparse_add_in_Z2(i + 1, i);
				return;
			}
			else
			{ // Case 3.2
				if (verbose)
					cout << "Case 3.2" << endl;
				RR.swapCols(i, i + 1);
				RR.swapRows(i, i + 1);
				UU.swapRows(i, i + 1);
				UU.swapCols(i, i + 1);

				matching[i] = i + 1;
				matching[i + 1] = i;
				double di = RR.bc[bci].second.second;
				double dii = RR.bc[bcii].second.second;
				double dmi, dmii;
				if (di == -1)
					dmi = -1;
				else
					dmi = matching[di];
				if (dii == -1)
					dmii = -1;
				else
					dmii = matching[dii];
				pair<int, int> pti(matching[RR.bc[bci].second.first], dmi);
				RR.bc[bci].second = pti;
				pair<int, int> ptii(matching[RR.bc[bcii].second.first], dmii);
				RR.bc[bcii].second = ptii;
				swap(RR.bc_inv[i], RR.bc_inv[i + 1]);

				return;
			}
		}

		else
		{ // Case 4
			if (verbose)
				cout << "Case 4" << endl;
			//	   pair<bool, int> u = U.entry(i, i + 1);
			if (u.first)
				UU.set_to_false(i, u.second);
			RR.swapCols(i, i + 1);
			RR.swapRows(i, i + 1);
			UU.swapRows(i, i + 1);
			UU.swapCols(i, i + 1);

			matching[i] = i + 1;
			matching[i + 1] = i;
			double di = RR.bc[bci].second.second;
			double dii = RR.bc[bcii].second.second;
			double dmi, dmii;
			if (di == -1)
				dmi = -1;
			else
				dmi = matching[di];
			if (dii == -1)
				dmii = -1;
			else
				dmii = matching[dii];
			pair<int, int> pti(matching[RR.bc[bci].second.first], dmi);
			RR.bc[bci].second = pti;
			pair<int, int> ptii(matching[RR.bc[bcii].second.first], dmii);
			RR.bc[bcii].second = ptii;
			swap(RR.bc_inv[i], RR.bc_inv[i + 1]);
			return;
		}
	}

	else
	{
		if (verbose)
			cout << "Case 5" << endl;
		RR.swapCols(i, i + 1);
		RR.swapRows(i, i + 1);
		UU.swapRows(i, i + 1);
		UU.swapCols(i, i + 1);

		matching[i] = i + 1;
		matching[i + 1] = i;
		double di = RR.bc[bci].second.second;
		double dii = RR.bc[bcii].second.second;
		double dmi, dmii;
		if (di == -1)
			dmi = -1;
		else
			dmi = matching[di];
		if (dii == -1)
			dmii = -1;
		else
			dmii = matching[dii];
		pair<int, int> pti(matching[RR.bc[bci].second.first], dmi);
		RR.bc[bci].second = pti;
		pair<int, int> ptii(matching[RR.bc[bcii].second.first], dmii);
		RR.bc[bcii].second = ptii;
		swap(RR.bc_inv[i], RR.bc_inv[i + 1]);

		return;
	}
};

/**
 * @brief Stores the variables to be able to apply a vineyard update
 * 
 */
class VineyardsPersistence
{

public:
	VineyardsPersistence() = default;

	VineyardsPersistence(boundary_matrix matrix, vector<int> dimensions, vector<double> filter, bool lower_star = false, bool sorted = false, bool verbose = false)
	{
		if (verbose)
			cout << "Creating matrix ..." << flush;
		SparseBoundaryMatrix R(matrix);
		R.dimensions = dimensions;
		SparseBoundaryMatrix U(matrix);
		pair<SparseBoundaryMatrix, SparseBoundaryMatrix> P(R, U);
		this->P = P;
		this->structure = matrix;
		this->lower_star = lower_star;
		this->sorted = sorted;
		if (!this->lower_star)
			this->filter = filter;
		else
		{
			get_lower_star_filter(filter);
			this->filter.swap(filter);
		}
		if (verbose)
			disp_vect(this->filter);
		int n = R.num_simplices();
		vector<int> perm(n);
		for (int i = 0; i < n; i++)
			perm[i] = i;
		this->permutation = perm;
		if (verbose)
			R.display();
		if (this->sorted)
		{
			this->swapped = false;
		}
		else
		{
			if (verbose)
				cout << "bubble sort" << endl;
			this->initial_sort();
			this->swapped = true;
		}
		if (verbose)
			cout << "sorted R" << endl;
		if (verbose)
			P.first.display();
		if (verbose)
			cout << " Done !" << endl;
	}

	VineyardsPersistence(boundary_matrix structure, boundary_matrix R, boundary_matrix U, vector<int> dimensions, vector<int> permutation, barcode bc, vector<int> bc_inv, bool lower_star = false)
	{
		SparseBoundaryMatrix RR(R);
		SparseBoundaryMatrix UU(U);
		RR.dimensions = dimensions;
		RR.bc = bc;
		RR.bc_inv = bc_inv;
		pair<SparseBoundaryMatrix, SparseBoundaryMatrix> P(RR, UU);
		this->P = P;
		this->structure = structure;
		this->lower_star = lower_star;
		this->permutation = permutation;
	}

	VineyardsPersistence(boundary_matrix structure, boundary_matrix R, boundary_matrix U, vector<int> dimensions, vector<int> permutation, barcode bc, vector<int> bc_inv, vector<int> row_map_R, vector<int> row_map_R_inv, vector<int> row_map_U, vector<int> row_map_U_inv, bool lower_star = false)
	{
		SparseBoundaryMatrix RR(R);
		SparseBoundaryMatrix UU(U);
		RR.dimensions = dimensions;
		RR.bc = bc;
		RR.bc_inv = bc_inv;
		RR.row_map = row_map_R;
		RR.row_map_inv = row_map_R_inv;
		UU.row_map = row_map_U;
		UU.row_map_inv = row_map_U_inv;
		pair<SparseBoundaryMatrix, SparseBoundaryMatrix> P(RR, UU);
		this->P = P;
		this->structure = structure;
		this->lower_star = lower_star;
		this->permutation = permutation;
	}

	/**
	 * @brief Computes the first barcode, and fills the variables in the class
	 * 
	 * @param verbose 
	 * @param debug 
	 */
	void initialize_barcode(bool verbose = true, bool debug = false)
	{
		if (verbose)
			cout << "Initializing barcode";
		auto elapsed = clock();
		pair<SparseBoundaryMatrix, SparseBoundaryMatrix> P = compute_barcode_sparse(this->P.first, this->swapped);
		this->P = P;
		// if (debug && !is_sorted(this->filter, P.first.dimensions))
		// {
		// 	cout << "??????????????????????????????????????" << endl;
		// 	P.first.display();
		// 	disp_vect(P.first.dimensions);
		// }
		if (verbose)
		{
			elapsed = clock() - elapsed;
			cout << "... Done ! It took " << ((float)elapsed) / CLOCKS_PER_SEC << " seconds." << endl;
		}
	}

	// TODO: use best sort possible
	/**
	 * @brief Sorts the boundary matrix wrt the filtration sorting. Linear if already sorted, but up to quadratic in general
	 * 
	 */
	void bubble_sort()
	{
		int n = this->filter.size();
		for (int i = n - 1; i > 0; i--)
		{
			bool sorted = true;
			for (int j = 0; j < i; j++)
			{
				if ((this->filter[j] > this->filter[j + 1]) || ((this->filter[j] == this->filter[j + 1]) && (this->P.first.dimensions[j] > this->P.first.dimensions[j + 1]))
					// 				&& (this->P.first.dimensions[j] == this->P.first.dimensions[j+1])
				)
				{
					swap(this->filter[j], this->filter[j + 1]);
					this->P.first.swapCols(j, j + 1);
					this->P.first.swapRows(j, j + 1);
					swap(this->permutation[j], this->permutation[j + 1]);
					sorted = false;
				}
			}
			if (sorted)
				break;
		}
	}

	/**
	 * @brief updates the persistence matrices according to a permutation
	 * 
	 * @param p 
	 */
	void update_rows_and_cols(permutation p)
	{
		const bool verbose = false;
		const bool debug = false;
		if (verbose)
		{
			cout << endl;
			disp_vect(p);
			cout << endl;
		}
		auto list_transpositions = permutation_to_transpositions(p);
		if (verbose)
		{
			cout << endl;
			disp_vect(list_transpositions);
			cout << endl;
		}
		if (debug)
		{
			disp_vect(disjoint_cycles(p));
			disp_vect(this->P.first.dimensions);
		}
		for (uint i = 0; i < list_transpositions.size(); i++)
		{
			this->P.first.swapCols(list_transpositions[i].first, list_transpositions[i].second);
			this->P.first.swapRows(list_transpositions[i].first, list_transpositions[i].second);
		}
		if (debug && !is_sorted(this->P.first.dimensions.begin(), this->P.first.dimensions.end()))
		{
			cout << " ?????" << endl;
			disp_vect(this->P.first.dimensions);
			cout << endl;
		}
	}

	void dimension_sort()
	{
		const bool verbose = false;
		if constexpr (verbose)
			cout << "DIMENSION SORT" << endl;
		if constexpr (verbose)
			disp_vect(P.first.dimensions);
		vector<int> p = sort_to_permutation(this->P.first.dimensions); // dimensions is sorted here
		this->P.first.dimensions = circ(this->P.first.dimensions, p);
		if constexpr (verbose)
			disp_vect(p);
		this->filter = circ(this->filter, p);
		this->permutation = circ(this->permutation, p);
		update_rows_and_cols(p);
		if (verbose)
			disp_vect(P.first.dimensions);
	}

	void initial_sort_dimension()
	{
		const bool verbose = false;
		vector<int> p = sort_to_permutation(this->P.first.dimensions);
		if (p.empty())
			return;
		this->filter = circ(this->filter, p);
		this->permutation = circ(this->permutation, p);
		update_rows_and_cols(p); // col swapping also updates dimension
		if constexpr (verbose)
		{
			cout << "Dimension sorted ?" << is_sorted(this->P.first.dimensions.begin(), this->P.first.dimensions.end()) << endl;
		}
	}

	void initial_sort_filter()
	{
		const bool verbose = false;
		vector<int> p = filter_dimension_to_permutation(this->filter, this->P.first.dimensions, false);
		if (p.empty())
			return;
		// 		this->filter = circ(this->filter,p);
		this->permutation = circ(this->permutation, p);
		update_rows_and_cols(p);
		if constexpr (verbose)
		{
			cout << "Dimension sorted ?" << is_sorted(this->P.first.dimensions.begin(), this->P.first.dimensions.end()) << endl;
		}
	}
	/**
	 * @brief Sorts according to dimension and filters the boundary_matrix.
	 * 
	 * @param verbose 
	 */
	void initial_sort(bool verbose = false)
	{
		initial_sort_dimension();
		initial_sort_filter();
	}

	// WARNING: this function assumes that filter and structure have the same order on vertices
	void get_lower_star_filter(vector<double> &filter)
	{
		int n = this->P.first.num_simplices();
		vector<double> good_filter(n);
		int count = 0;
		for (int i = 0; i < n; i++)
		{
			boundary L = this->structure[i];
			if (L.size() == 0)
			{
				good_filter[i] = filter[count];
				count += 1;
			}
			else
			{
				good_filter[i] = good_filter[L[0]];
				int dim = L.size();
				for (int d = 1; d < dim; d++)
					good_filter[i] = max(good_filter[i], good_filter[L[d]]);
			}
		}
		//	   return good_filter;
		filter.swap(good_filter);
	}

	void update_old(vector<double> &new_filter)
	{
		const bool verbose = false;
		int n = this->P.first.num_simplices();
		if (this->lower_star)
		{
			// 		  vector<double> ls_filter(n);
			this->get_lower_star_filter(new_filter);
			// 		  new_filter.swap(ls_filter);
		}
		//	   vector<double> permuted_filter = new_filter;
		for (int i = 0; i < n; i++)
			this->filter[i] = new_filter[this->permutation[i]];
		//	   this->filter = permuted_filter;
		// for(int i = 0; i < new_filter.size(); i++)  cout << this->P.first.dimensions[i] << " "; cout << endl;
		// this->P.first.display();
		uint k = 0;

		for (int i = n - 1; i > 0; i--)
		{
			bool sorted = true;
			for (int j = 0; j < i; j++)
			{
				if (
					(this->P.first.dimensions[j] == this->P.first.dimensions[j + 1]) && ((this->filter[j] > this->filter[j + 1]))
					// 				|| ((this->filter[j] == this->filter[j+1]) && (this->P.first.dimensions[j] > this->P.first.dimensions[j+1]))
				)
				{
					swap(this->permutation[j], this->permutation[j + 1]);
					vineyard_update(j, this->P, false);
					swap(this->filter[j], this->filter[j + 1]);
					sorted = false;
					if constexpr (verbose)
						k++;
				}
			}
			if (sorted)
				break; // for(unsigned int i = 0; i < this->filter.size(); i++)  cout << this->filter[i] << " "; cout << endl;
		}
		if constexpr (verbose)
			cout << "Permuted " << k << "times, with " << n << " simplices." << endl;
	};

	void update(vector<double> &new_filter) { update_old(new_filter); }

	void permutation_update(const permutation &p)
	{
		const bool verbose = false;
		const bool debug = false;
		// 		this->filter = circ(this->filter,p);
		this->permutation = circ(this->permutation, p);

		if (verbose && debug)
		{
			auto list = disjoint_cycles(p);
			if (list.size() > 0)
			{
				cout << "Permutation update : ";
				disp_vect(disjoint_cycles(p));
				cout << endl;
			}
			int a;
			cin >> a;
		}
		if (verbose)
			cout << endl;
		vector<int> list_transpositions = coxeter(p, false);

		for (uint i = 0; i < list_transpositions.size(); i++)
		{
			// Verbose stuff
			if (verbose)
			{
				cout << "Step " << i + 1 << "/" << list_transpositions.size() << ". Transposition " << list_transpositions[i] << " over " << p.size() << flush;
				// 				cout<< " with transpositions ("; disp_vect(list_transpositions); cout <<")	 "<< flush;
			}
			vineyard_update(list_transpositions[i], this->P);
			// 			this->P = vineyard_update(list_transpositions[i], this->P);
			// 			vineyard_iteration(list_transpositions[i], this->P);
			if (verbose)
				cout << "\r";
		}
		if (verbose && !list_transpositions.empty())
			cout << endl;
	}

	// From a filter of the 0 skeleton, updates this->filter as the lower_star filter of new_filter.
	// It's the same as get_lower_star_filter, but with less memory calls.
	// WARNING it assumes that this-> filter is of size number_of_simplicies (this is done by the VineyardsPersistence constructor anyway).
	// WARNING Needs permutation_inverse otherwise its much slower...
	// DONOTUSE
	void update_lower_start_filter(vector<double> new_filter)
	{
		for (uint i = 0; i < this->P.first.num_simplices(); i++)
		{
			// As it is sorted by dimension, we will always pass by simplices dimension by dimension (and thus reaching boundary => we already reached boundary simplicies) even if we permute with this->permutation.
			uint dim = this->structure[i].size(); // if simplex i has no boundary -> dimension 0, other wise dimension dim-1.
			vector<double> p_inv(P.first.num_simplices());
			for (uint k = 1; k < P.first.num_simplices(); k++)
				p_inv[this->permutation[k]] = k;
			if (dim == 0)
			{
				this->filter[i] = new_filter[this->permutation[i]];
				continue;
			}
			this->filter[i] = this->filter[p_inv[this->structure[this->permutation[i]][0]]];
			for (uint j = 1; j < dim; j++)
			{
				this->filter[this->permutation[i]] = max(this->filter[i], this->filter[p_inv[this->structure[this->permutation[i]][j]]]);
			}
		}
	}

	/**
	 * @brief updates the RU decomposition with @p new_filter, via a coxeter decomposition.
	 * 
	 * @param new_filter 
	 */
	void RU_filter_update(vector<double> &new_filter)
	{
		const bool verbose = false;
		if (this->lower_star)
			get_lower_star_filter(new_filter); // generate full filter from lower_star

		uint n = new_filter.size();
		for (uint i = 0; i < n; i++)
			this->filter[i] = new_filter[this->permutation[i]]; // updates filtration, with old permutation

		vector<int> p = filter_dimension_to_permutation(this->filter, this->P.first.dimensions);
		if (p.empty()) // in that case the permutation is the identity, we don't have to do anything
			return;
		permutation_update(p); // Computes the vineyard update associated with p, and update permutation

		if constexpr (verbose && !is_sorted(this->filter, this->P.first.dimensions))
		{
			cout << "????????????????????????????????" << endl;
			disp_vect(disjoint_cycles(p));
			cout << " is : ";
			disp_vect(p);
			cout << endl;
			disp_vect(P.first.dimensions);
			cout << endl;
		}
	}

	/**
	 * @brief Gets diagram from RU decomposition.
	 * 
	 */
	void get_diagram()
	{
		//	   this->dgm.clear();
		uint nbc = this->P.first.bc.size();
		if (dgm.size() != nbc)
			dgm.resize(nbc);
		for (uint i = 0; i < nbc; i++)
		{
			double b = this->filter[this->P.first.bc[i].second.first];
			double d;
			int dd = this->P.first.bc[i].second.second;
			if (dd != -1)
				d = this->filter[dd];
			else
				d = DBL_MAX;
			pair<double, double> pt(b, d);
			// 		this->dgm.push_back(pair<int,pair<double,double>>(P.first.bc[i].first, pt));
			dgm[i].first = P.first.bc[i].first;
			dgm[i].second = pt;
		}
	}

	void display_diagram()
	{
		for (unsigned int i = 0; i < this->dgm.size(); i++)
			cout << this->dgm[i].first << " " << this->dgm[i].second.first << " " << this->dgm[i].second.second << endl;
	}
	void display_filt()
	{
		for (unsigned int i = 0; i < this->filter.size(); i++)
			cout << this->filter[i] << " ";
		cout << endl;
	}
	vector<int> permutation;
	vector<pair<int, pair<double, double>>> dgm;
	vector<double> filter;
	bool lower_star;
	bool swapped;
	bool sorted;
	pair<SparseBoundaryMatrix, SparseBoundaryMatrix> P;
	boundary_matrix structure;
};

#endif // VINEYARDS_H_INCLUDED
