#ifndef MULTIPERS_SCC_IO_H
#define MULTIPERS_SCC_IO_H

#include <cassert>
#include <cstddef>
#include <ostream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <iomanip>
#include <cmath>

// #include "truc.h"

// using Gudhi::multiparameter::interface::Truc;

bool is_comment_or_empty_line(const std::string& line) {
  size_t current = line.find_first_not_of(' ', 0);
  if (current == std::string::npos) return true;  // is empty line
  if (line[current] == '#') return true;  // is comment
  return false;
}

template <class Slicer>
Slicer read_scc_file(std::string inFilePath, bool isRivetCompatible = false){
  std::string line;
  std::ifstream file(inFilePath);
  Slicer slicer;

  auto error = [&file](std::string msg){
    file.close();
    throw std::invalid_argument(msg);
  };

  if (file.is_open()) {
    while (getline(file, line, '\n') && is_comment_or_empty_line(line));
    if (!file) error("Empty file!");

    if (isRivetCompatible && line.compare("firep") != 0) error("Wrong file format. Should start with 'firep'.");
    if (!isRivetCompatible && line.compare("scc2020") != 0) error("Wrong file format. Should start with 'scc2020'.");

    while (getline(file, line, '\n') && is_comment_or_empty_line(line));
    if (!file) error("Premature ending of the file. Stops before numbers of parameters.");
    


    // std::vector<ID_handle> data;
    // unsigned int id = 0;
    // double timestamp;
    // lineType type;

    // while (getline(file, line, '\n') && read_operation(line, data, timestamp) == COMMENT);
    // double lastTimestamp = timestamp;
    // // first operation has to be an insertion.
    // zp.insert_face(id, data, 0, timestamp);

    // while (getline(file, line, '\n')) {
    //   type = read_operation(line, data, timestamp);
    //   if (type != COMMENT && lastTimestamp != timestamp) {
    //     lastTimestamp = timestamp;
    //   }

    //   if (type == INCLUSION) {
    //     ++id;
    //     int dim = data.size() == 0 ? 0 : data.size() - 1;
    //     zp.insert_face(id, data, dim, timestamp);
    //   } else if (type == REMOVAL) {
    //     ++id;
    //     zp.remove_face(data[0], timestamp);
    //   }
    // }

    file.close();
  } else {
    std::cerr << "Unable to open input file." << std::endl;
    file.setstate(std::ios::failbit);
  }

  return slicer;
};

template <class Slicer>
void write_scc_file(std::string outFilePath,
                    const Slicer& slicer,
                    int numberOfParameters = -1,
                    int degree = -1,
                    bool rivetCompatible = false,
                    bool IgnoreLastGenerators = false,
                    bool stripComments = false,
                    bool reverse = false) {
  if (numberOfParameters < 0){
    numberOfParameters = slicer.num_parameters();
  }
  assert(numberOfParameters > 0 && "Invalid number of parameters!");

  std::ofstream file(outFilePath);

  if (rivetCompatible) file << "firep\n";
  else file << "scc2020\n";

  if (!stripComments && !rivetCompatible) file << "# Number of parameters\n";

  if (rivetCompatible){
    assert(numberOfParameters == 2 && "Rivet only handles bifiltrations.");
    file << "Filtration 1\n";
    file << "Filtration 2\n";
  } else {
    file << std::to_string(numberOfParameters) << "\n";
  }

  if (!stripComments) file << "# Sizes of generating sets\n";

  using Filtration_value = typename Slicer::Filtration_value;

  auto& boundaries = slicer.structure;
  int maxDim = boundaries.max_dimension();
  int minDim = maxDim;

  std::vector<std::vector<std::size_t> > indicesByDim(maxDim + 1);
  std::vector<std::size_t> shiftedIndices(boundaries.size());
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    auto dim = boundaries.dimension(i);
    minDim = dim < minDim ? dim : minDim;
    auto& atDim = indicesByDim[reverse ? dim : maxDim - dim];
    shiftedIndices[i] = atDim.size();
    atDim.push_back(i);
  }
  if (degree < 0) degree = minDim;
  int minIndex = reverse ? degree - 1 : 0;
  int maxIndex = reverse ? maxDim : maxDim - degree + 1;
  if (maxIndex < -1) maxIndex = -1;
  if (rivetCompatible || IgnoreLastGenerators) maxIndex--;

  auto print_fil_values = [&](const Filtration_value& fil) {
    if (fil.is_finite()) {
      assert(fil.size() == static_cast<unsigned int>(numberOfParameters));
      for (auto f : fil) file << f << " ";
    } else {
      assert(fil.size() == 1);
      for (int p = 0; p < numberOfParameters; ++p) file << fil[0] << " ";
    }
  };

  if (minIndex < 0) file << 0 << " ";
  for (int i = 0; i < minIndex; ++i) file << 0 << " ";
  for (int i = std::max(minIndex, 0); i <= std::min(maxDim, maxIndex); ++i) {
    file << indicesByDim[i].size() << " ";
  }
  for (int i = maxIndex + 1; i <= maxDim; ++i) file << 0 << " ";
  if (maxIndex > maxDim) file << 0;
  file << "\n";

  file << std::setprecision(std::numeric_limits<typename Filtration_value::value_type>::digits10 + 1);

  std::size_t startIndex = reverse ? minIndex + 1 : minIndex;
  std::size_t endIndex = reverse ? maxIndex : maxIndex - 1;
  const auto& filtValues = slicer.get_filtrations();
  int currDim;
  if (reverse) currDim = minIndex == -1 ? 0 : minIndex;
  else currDim = maxIndex == maxDim + 1 ? maxDim + 1 : maxDim;

  if (reverse){
    if (!stripComments) file << "# Block of dimension " << currDim++ << "\n";
    if (minIndex >= 0) {
      for (auto index : indicesByDim[minIndex]) {
        print_fil_values(filtValues[index]);
        file << ";\n";
      }
    }
  }
  for (std::size_t i = startIndex; i <= endIndex; ++i) {
    if (!stripComments) {
      file << "# Block of dimension " << currDim << "\n";
      if (reverse) ++currDim;
      else --currDim;
    }
    for (auto index : indicesByDim[i]) {
      print_fil_values(filtValues[index]);
      file << "; ";
      for (auto b : boundaries[index]) file << shiftedIndices[b] << " ";
      file << "\n";
    }
  }
  if (!reverse){
    if (!stripComments) file << "# Block of dimension " << currDim << "\n";
    if (maxIndex <= maxDim) {
      for (auto index : indicesByDim[maxIndex]) {
        print_fil_values(filtValues[index]);
        file << ";\n";
      }
    }
  }
};

// enum lineType : int { INCLUSION, REMOVAL, COMMENT };

// lineType read_operation(std::string& line, std::vector<ID_handle>& faces, double& timestamp) {
//   lineType type;
//   faces.clear();
//   ID_handle num;

//   size_t current = line.find_first_not_of(' ', 0);
//   if (current == std::string::npos) return COMMENT;

//   if (line[current] == 'i')
//     type = INCLUSION;
//   else if (line[current] == 'r')
//     type = REMOVAL;
//   else if (line[current] == '#')
//     return COMMENT;
//   else {
//     std::clog << "(1) Syntaxe error in file." << std::endl;
//     exit(0);
//   }

//   current = line.find_first_not_of(' ', current + 1);
//   if (current == std::string::npos) {
//     std::clog << "(2) Syntaxe error in file." << std::endl;
//     exit(0);
//   }
//   size_t next = line.find_first_of(' ', current);
//   timestamp = std::stod(line.substr(current, next - current));

//   current = line.find_first_not_of(' ', next);
//   while (current != std::string::npos) {
//     next = line.find_first_of(' ', current);
//     num = std::stoi(line.substr(current, next - current));
//     faces.push_back(num);
//     current = line.find_first_not_of(' ', next);
//   }

//   return type;
// }

// //example of input file: example/zigzag_filtration_example.txt
// int main(int argc, char* const argv[]) {
//   if (argc != 2) {
//     if (argc < 2)
//       std::clog << "Missing argument: input file name is needed." << std::endl;
//     else
//       std::clog << "Too many arguments: only input file name is needed." << std::endl;
//     return 0;
//   }

//   std::string line;
//   std::ifstream file(argv[1]);

//   //std::cout could be replaced by any other output stream
//   ZP zp([](Dimension dim, Filtration_value birth, Filtration_value death) {
//     std::cout << "[" << dim << "] ";
//     std::cout << birth << " - " << death;
//     std::cout << std::endl;
//   });

//   if (file.is_open()) {
//     std::vector<ID_handle> data;
//     unsigned int id = 0;
//     double timestamp;
//     lineType type;

//     while (getline(file, line, '\n') && read_operation(line, data, timestamp) == COMMENT);
//     double lastTimestamp = timestamp;
//     // first operation has to be an insertion.
//     zp.insert_face(id, data, 0, timestamp);

//     while (getline(file, line, '\n')) {
//       type = read_operation(line, data, timestamp);
//       if (type != COMMENT && lastTimestamp != timestamp) {
//         lastTimestamp = timestamp;
//       }

//       if (type == INCLUSION) {
//         ++id;
//         int dim = data.size() == 0 ? 0 : data.size() - 1;
//         zp.insert_face(id, data, dim, timestamp);
//       } else if (type == REMOVAL) {
//         ++id;
//         zp.remove_face(data[0], timestamp);
//       }
//     }

//     file.close();
//   } else {
//     std::clog << "Unable to open input file." << std::endl;
//     file.setstate(std::ios::failbit);
//   }

//   //retrieve infinite bars remaining at the end
//   //again std::cout could be replaced by any other output stream
//   zp.get_current_infinite_intervals([](Dimension dim, Filtration_value birth) {
//     std::cout << "[" << dim << "] ";
//     std::cout << birth << " - inf";
//     std::cout << std::endl;
//   });

//   return 0;
// }

#endif // MULTIPERS_SCC_IO_H