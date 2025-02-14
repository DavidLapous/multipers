#ifndef MULTIPERS_SCC_IO_H
#define MULTIPERS_SCC_IO_H

#include <algorithm>
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

template <class Slicer>
inline Slicer read_scc_file(const std::string& inFilePath,
                            bool isRivetCompatible = false,
                            bool isReversed = false,
                            int shiftDimensions = 0) {
  using Filtration_value = typename Slicer::Filtration_value;

  std::string line;
  std::ifstream file(inFilePath);
  Slicer slicer;
  unsigned int numberOfParameters;

  if (file.is_open()) {
    auto error = [&file](std::string msg) {
      file.close();
      throw std::invalid_argument(msg);
    };
    auto is_comment_or_empty_line = [](const std::string& line) -> bool {
      size_t current = line.find_first_not_of(' ', 0);
      if (current == std::string::npos) return true;  // is empty line
      if (line[current] == '#') return true;          // is comment
      return false;
    };

    while (getline(file, line, '\n') && is_comment_or_empty_line(line));
    if (!file) error("Empty file!");

    if (isRivetCompatible && line.compare("firep") != 0) error("Wrong file format. Should start with 'firep'.");
    if (!isRivetCompatible && line.compare("scc2020") != 0) error("Wrong file format. Should start with 'scc2020'.");

    while (getline(file, line, '\n') && is_comment_or_empty_line(line));
    if (!file) error("Premature ending of the file. Stops before numbers of parameters.");

    if (isRivetCompatible) {
      numberOfParameters = 2;
      getline(file, line, '\n');  // second rivet label
    } else {
      std::size_t current = line.find_first_not_of(' ', 0);
      std::size_t next = line.find_first_of(' ', current);
      numberOfParameters = std::stoi(line.substr(current, next - current));
    }

    while (getline(file, line, '\n') && is_comment_or_empty_line(line));
    if (!file) error("Premature ending of the file. Not a single cell was specified.");

    std::vector<unsigned int> counts;
    unsigned int numberOfCells = 0;
    counts.reserve(line.size() + shiftDimensions);
    std::size_t current = line.find_first_not_of(' ', 0);
    if (shiftDimensions != 0 && isReversed && current != std::string::npos) {
      if (shiftDimensions > 0) {
        counts.resize(shiftDimensions, 0);
      } else {
        for (int i = shiftDimensions; i < 0 && current != std::string::npos; ++i) {
          std::size_t next = line.find_first_of(' ', current);
          current = line.find_first_not_of(' ', next);
        }
      }
    }
    while (current != std::string::npos) {
      std::size_t next = line.find_first_of(' ', current);
      counts.push_back(std::stoi(line.substr(current, next - current)));
      numberOfCells += counts.back();
      current = line.find_first_not_of(' ', next);
    }
    if (shiftDimensions != 0 && !isReversed) {
      counts.resize(counts.size() + shiftDimensions, 0);
    }

    std::size_t dimIt = 0;
    while (dimIt < counts.size() && counts[dimIt] == 0) ++dimIt;

    if (dimIt == counts.size()) return slicer;

    std::size_t shift = isReversed ? 0 : counts[dimIt];
    unsigned int nextShift = isReversed ? 0 : counts.size() == 1 ? 0 : counts[dimIt + 1];
    unsigned int tmpNextShift = counts[dimIt];

    auto get_boundary = [&isReversed, &numberOfCells](const std::string& line,
                                                      std::size_t start,
                                                      std::size_t shift) -> std::vector<unsigned int> {
      std::vector<unsigned int> res;
      res.reserve(line.size() - start);
      std::size_t current = line.find_first_not_of(' ', start);
      while (current != std::string::npos) {
        std::size_t next = line.find_first_of(' ', current);
        unsigned int idx = std::stoi(line.substr(current, next - current)) + shift;
        res.push_back(isReversed ? idx : numberOfCells - 1 - idx);
        current = line.find_first_not_of(' ', next);
      }
      std::sort(res.begin(), res.end());
      return res;
    };
    auto get_filtration_value = [](const std::string& line, std::size_t end) -> Filtration_value {
      Filtration_value res(0);
      res.reserve(end);
      bool isPlusInf = true;
      bool isMinusInf = true;
      std::size_t current = line.find_first_not_of(' ', 0);
      while (current < end) {
        std::size_t next = line.find_first_of(' ', current);
        res.push_back(std::stod(line.substr(current, next - current)));
        if (isPlusInf && res.back() != Filtration_value::T_inf) isPlusInf = false;
        if (isMinusInf && res.back() != -Filtration_value::T_inf) isMinusInf = false;
        current = line.find_first_not_of(' ', next);
      }
      if (isPlusInf) res = Filtration_value::inf();
      if (isMinusInf) res = Filtration_value::minus_inf();
      return res;
    };

    std::vector<std::vector<unsigned int> > generator_maps(numberOfCells);
    std::vector<int> generator_dimensions(numberOfCells);
    std::vector<typename Slicer::Filtration_value> generator_filtrations(numberOfCells);
    std::size_t i = 0;
    // because of possible negative dimension shifts, the document should not always be read to the end
    // therefore `dimIt < counts.size()` is also a stop condition
    while (getline(file, line, '\n') && dimIt < counts.size()) {
      if (!is_comment_or_empty_line(line)) {
        std::size_t sep = line.find_first_of(';', 0);
        generator_filtrations[i] = get_filtration_value(line, sep);
        if (generator_filtrations[i].is_finite() && generator_filtrations[i].num_parameters() != numberOfParameters)
          error("Wrong format. The number of parameters does not match.");
        generator_maps[i] = get_boundary(line, sep + 1, shift);
        generator_dimensions[i] = isReversed ? dimIt : counts.size() - 1 - dimIt;

        --counts[dimIt];
        while (dimIt < counts.size() && counts[dimIt] == 0) {
          ++dimIt;
          if (dimIt != counts.size()) {
            shift += nextShift;
            nextShift = isReversed ? tmpNextShift : dimIt < counts.size() - 1 ? counts[dimIt + 1] : 0;
            tmpNextShift = counts[dimIt];
          }
        }
        ++i;
      }
    }

    if (!isReversed) {  // to order by dimension
      std::reverse(generator_dimensions.begin(), generator_dimensions.end());
      std::reverse(generator_maps.begin(), generator_maps.end());
      std::reverse(generator_filtrations.begin(), generator_filtrations.end());
    }

    slicer = Slicer(generator_maps, generator_dimensions, generator_filtrations);

    file.close();
  } else {
    std::cerr << "Unable to open input file: " << inFilePath << std::endl;
    file.setstate(std::ios::failbit);
  }

  return slicer;
};

template <class Slicer>
inline void write_scc_file(const std::string& outFilePath,
                           const Slicer& slicer,
                           int numberOfParameters = -1,
                           int degree = -1,
                           bool rivetCompatible = false,
                           bool IgnoreLastGenerators = false,
                           bool stripComments = false,
                           bool reverse = false) {
  constexpr bool verbose = false;
  if (numberOfParameters < 0) {
    numberOfParameters = slicer.num_parameters();
  }
  assert(numberOfParameters > 0 && "Invalid number of parameters!");

  std::ofstream file(outFilePath);

  if (rivetCompatible)
    file << "firep\n";
  else
    file << "scc2020\n";

  if (!stripComments && !rivetCompatible)
    file << "# This file was generated by multipers (https://github.com/DavidLapous/multipers).\n";

  if (!stripComments && !rivetCompatible) file << "# Number of parameters\n";

  if (rivetCompatible) {
    assert(numberOfParameters == 2 && "Rivet only handles bifiltrations.");
    file << "Filtration 1\n";
    file << "Filtration 2\n";
  } else {
    file << std::to_string(numberOfParameters) << "\n";
  }

  if (!stripComments) file << "# Sizes of generating sets\n";

  using Filtration_value = typename Slicer::Filtration_value;

  const auto& boundaries = slicer.get_structure();
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
  if (IgnoreLastGenerators) maxIndex--;
  if (rivetCompatible)
    minIndex = maxIndex -2;

  if constexpr (verbose) {
    std::cout << "minDim = " << minDim << " maxDim = " << maxDim << " minIndex = " << minIndex
              << " maxIndex = " << maxIndex << " degree = " << degree << std::endl;
  }

  auto print_fil_values = [&](const Filtration_value& fil) {
    if (fil.is_finite()) {
      if constexpr (Filtration_value::is_multicritical()) {
        for (const auto& ifil : fil) {
          for (auto f : ifil) file << f << " ";
        }
      } else {
        assert(fil.size() == static_cast<unsigned int>(numberOfParameters));
        for (auto f : fil) file << f << " ";
      }
    } else {
      // assert(fil.size() == 1);
      for (int p = 0; p < numberOfParameters; ++p) file << fil[0] << " ";
    }
  };

  if (minIndex < 0) file << 0 << " ";
  for (int i = 0; i < minIndex; ++i) file << 0 << " ";
  for (int i = std::max(minIndex, 0); i <= std::min(maxDim, maxIndex); ++i) {
    file << indicesByDim[i].size() << " ";
  }
  if (!rivetCompatible)
    for (int i = maxIndex + 1; i <= maxDim; ++i) file << 0 << " ";
  if (maxIndex > maxDim) file << 0;
  file << "\n";

  file << std::setprecision(std::numeric_limits<typename Filtration_value::value_type>::digits);

  std::size_t startIndex = reverse ? minIndex + 1 : minIndex;
  std::size_t endIndex = reverse ? maxIndex : maxIndex - 1;
  const auto& filtValues = slicer.get_filtrations();
  int currDim;
  if (reverse)
    currDim = minIndex == -1 ? 0 : minIndex;
  else
    currDim = maxIndex == maxDim + 1 ? maxDim + 1 : maxDim;

  if (reverse) {
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
      if (reverse)
        ++currDim;
      else
        --currDim;
    }
    for (auto index : indicesByDim[i]) {
      print_fil_values(filtValues[index]);
      file << "; ";
      for (auto b : boundaries[index]) file << shiftedIndices[b] << " ";
      file << "\n";
    }
  }
  if (!reverse) {
    if (!stripComments) file << "# Block of dimension " << currDim << "\n";
    if (maxIndex <= maxDim) {
      for (auto index : indicesByDim[maxIndex]) {
        print_fil_values(filtValues[index]);
        file << ";\n";
      }
    }
  }
};

#endif  // MULTIPERS_SCC_IO_H
