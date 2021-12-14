#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstdlib>
#include <cfloat>
#include "vineyards.h"

int main(int argc, char *argv[]){

  string complex_file = argv[1]; string values_file = argv[2]; string output_file = argv[3]; int essential = atoi(argv[4]); string line; 

  int max_dim = 0;
  ifstream in_complex_file(complex_file); boundary_matrix structure; vector<int> dims; map<vector<int>, int> positions; int ID = 0;
  while(getline(in_complex_file, line)){
    stringstream s(line); int vertex; boundary bd, matrix_bd; while(s >> vertex) bd.push_back(vertex); 
    if(bd.size() == 1){structure.push_back(matrix_bd); dims.push_back(bd.size()-1); }
    else{
      for(int i = 0; i < bd.size(); i++){vector<int> new_bd = bd; new_bd.erase(new_bd.begin() + i); matrix_bd.push_back(positions[new_bd]);}
      structure.push_back(matrix_bd); int bds = bd.size()-1; dims.push_back(bds); max_dim = max(max_dim, bds);
    }
    positions.insert(pair<vector<int>, int>(bd, ID)); ID += 1;
    
  }

  /*cout << "read complex" << endl;
  cout << "dimensions: ";
  for (int i = 0; i < dims.size(); i++)  cout << dims[i] << " "; cout << endl;
  for (int i = 0; i < structure.size(); i++){
    for (int j = 0; j < structure[i].size(); j++)  cout << structure[i][j] << " ";
    cout << endl;
  }*/

  if(essential) max_dim += 1;
  VineyardsPersistence VP; ifstream in_values_file(values_file); bool first = true; int time = 0; 

  vector<shared_ptr<ofstream> > out_files;
  for (int d = 0; d < max_dim; d++){
    shared_ptr<ofstream> out(new ofstream);
    string sdim = to_string(d); string ext = ".vin";
    string fileName = output_file + sdim + ext;
    out->open(fileName.c_str());
    out_files.push_back(out);
  }

  
  vector<vector<pair<pair<int,int>,pair<double,double>>>> vineyards;
  while(getline(in_values_file, line)){
    //cout << "line " << time << ": ";
    stringstream s(line); double val; vector<double> filt; while(s >> val) filt.push_back(val);
    if(first){
      VineyardsPersistence VPP(structure, dims, filt, true, false); VPP.initialize_barcode(); vector<pair<pair<int,int>,pair<double,double>>> tmp; for(int i = 0; i < VPP.P.first.bc.size(); i++)  vineyards.push_back(tmp);
      //cout << "first barcode computed" << endl;
      VP = VPP; first = false;
    }
    else  VP.update(filt);
    VP.get_diagram();
    for (int d = 0; d < max_dim; d++){
      for (int i = 0; i < VP.dgm.size(); i++){
        if(VP.dgm[i].first == d){pair<double,double> pt(VP.dgm[i].second.first, VP.dgm[i].second.second); vineyards[i].push_back(pair<pair<int,int>,pair<double,double>>(pair<int,int>(time,d),pt));}
      }
    }
    time += 1;
  }
  
  for (int d = 0; d < max_dim; d++){
    for (int i = 0; i < vineyards.size(); i++){
      vector<pair<pair<int,int>,pair<double,double>>> vine = vineyards[i]; int Lv = vine.size();
      bool trivial = true; for (unsigned int j = 0; j < Lv; j++){if(vine[j].second.second != vine[j].second.first){trivial = false; break;}}
      if (!trivial){
      for (unsigned int j = 0; j < Lv; j++){
        if(vine[j].first.second == d){
          if(vine[j].second.second == DBL_MAX){if(essential) *out_files[d] << vine[j].second.first << " inf " << vine[j].first.first << " ";}
          else{*out_files[d] << vine[j].second.first << " " << vine[j].second.second << " " << vine[j].first.first << " ";}
        }
      }}
      *out_files[d] << endl;
    }
  }
  
  return 0;
}
