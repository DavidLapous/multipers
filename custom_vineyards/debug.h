/**
 * @file debug.h
 * @author David Loiseaux
 * @brief Display functions for debug purposes
 * 
 * @copyright Copyright (c) 2021 Inria
 * 
 */


#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include "dependences.h"


void disp(point_2 pt){
	cout << "(" << pt.first << ", " << pt.second << ")";
}

void disp(vector<barcoded> barcodes){
	for(uint i=0; i< barcodes.size();i++){
		for(uint j=0; j < barcodes[0].size(); j++ ){
			cout <<barcodes[i][j].first << "-(" << barcodes[i][j].second.first << ", " << barcodes[i][j].second.second << ") ";
		}
		cout << endl;

	}
}


template<typename T>
void disp_vect(vector<T> v){
	for(uint i=0; i< v.size(); i++){
		cout << v[i] << " ";
	}
	cout <<endl;
}

template<typename T>
void disp_vect(list<T> v){
	while(!v.empty()){
		cout << v.front() << " ";
		v.pop_front();
	}
	cout <<endl;
}

template<typename T>
void disp_vect(vector<pair<T,T>> v){
	for(uint i=0; i< v.size(); i++){
		cout << "(" << v[i].first << " " << v[i].second <<")  ";
	}
}


template<typename T>
void disp_vect(vector<vector<T>> v, bool show_small = true){
	for(uint i=0; i< v.size(); i++){
		if(v[i].size()<=1 && !show_small) continue;
		cout << "(";
		for (uint j=0; j<v[i].size();j++){
			cout << v[i][j];
			if(j < v[i].size()-1) cout << " ";
		}
		cout << ") ";
	}
	cout << endl;
}

template<typename T>
void disp_vect2(vector<vector<T>> v){
	for(uint i=0; i< v.size(); i++){
		disp_vect(v[i]);
		cout << endl;
	}
}






#endif // DEBUG_H_INCLUDED
