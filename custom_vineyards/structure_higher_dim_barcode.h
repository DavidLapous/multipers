/**
 * @file structure_higher_dim_barcode.h
 * @author David Loiseaux
 * @brief Structures to handle higher dimensional persistence=
 * 
 * @copyright Copyright (c) 2021 Inria
 * 
 */


#ifndef STRUCTURE_HIGHER_DIM_BARCODE_H_INCLUDED
#define STRUCTURE_HIGHER_DIM_BARCODE_H_INCLUDED

#include "dependences.h"

template<typename T>
struct Tree{
	T node;
	vector<Tree<T>> childs;
	Tree<T>(T node, vector<Tree<T>> childs){
		this -> node = node;
		this -> childs = childs;
	}
	Tree<T>(){};
};

template<uint dimension, typename T>
struct Matrix_nd{
	vector<Matrix_nd<dimension-1,T>> matrix;
	uint number_of_cells;
	Matrix_nd(){
		number_of_cells = 0;
	}
	Matrix_nd(const vector<uint> &n, uint start=0){
		assert(start < n.size());
		this->matrix.resize(n[start]);
		number_of_cells =1;
		for(uint j=0;j<n[start];j++){
			this->matrix[j] = Matrix_nd<dimension-1,T>(n, start+1);
			number_of_cells*=n[j];
		}

	}
	Matrix_nd<dimension-1,T>& operator [] (uint indice) {
		return matrix[indice];
	};
	void display() const{
		for(const auto &truc : this->matrix){
			truc.display();
		}
		cout << endl;
	}
	void set(vector<uint> indice, T value, uint start = 0){
		matrix[indice[start]].set(indice, value, start+1);
	}
	void swap(vector<uint> indice, T to_swap, uint start=0){
		matrix[indice[start]].swap(indice,to_swap, start+1);
	}

	void to_vector(vector<T> &container){
		container.resize(number_of_cells);
	}
};
template<typename T>
struct Matrix_nd<0,T>{
	uint number_of_cells;
	Matrix_nd(){
		number_of_cells = 0;
	}
};

template<typename T>
struct Matrix_nd<1,T>{
	vector<T> matrix;
	uint number_of_cells;
	Matrix_nd(vector<T>& v){
		number_of_cells = v.size();
		this->matrix.swap(v);

	}
	Matrix_nd(){};
	Matrix_nd(uint n){
		number_of_cells = n;
		this->matrix.resize(n);
	}
	Matrix_nd(const vector<uint> &n, uint start=0){
		assert(start < n.size());
		number_of_cells = n[start];
		this->matrix.resize(n[start]);
	}

	T& operator [] (uint indice) {
		return matrix[indice];
	};
	void display() const{
		disp_vect(this->matrix);
	}
	void set(uint indice, T value){
		this->matrix[indice] = value;
	}
	void set(vector<uint> indice, T value, uint start = 0){
		this->matrix[indice[start]] = value;
	}
	void swap(vector<T> &to_swap){
		matrix.swap(to_swap);
	}
	void swap(vector<uint> indice, T to_swap, uint start=0){
		matrix[indice[start]].swap(to_swap);
	}
};


uint position_size_to_indice(const vector<uint> &position, const vector<uint> &size){
	uint indice = 0;
	assert(position.size() == size.size() && "Position and Size vector must be of the same size !");
	uint last_product=1;
	for (uint i=0; i<position.size(); i++){
		indice += last_product*position[i];
		last_product *= size[i];
	}
	return indice;
}



#endif // STRUCTURE_HIGHER_DIM_BARCODE_H_INCLUDED
