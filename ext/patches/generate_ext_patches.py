#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = Path(__file__).resolve().parent


FUNCTION_DELAUNAY_LOG_UTILS = r"""#pragma once

#include <iostream>

#include "ext_interface/backend_log_policy.hpp"

namespace function_delaunay {

class null_streambuf : public std::streambuf {
 protected:
  int overflow(int c) override { return traits_type::not_eof(c); }
};

inline std::ostream& log_stream() {
  if (multipers::backend_log_policy::backend_log_enabled(
          multipers::backend_log_policy::backend_log_bit::function_delaunay)) {
    return std::cout;
  }
  static null_streambuf buffer;
  static std::ostream out(&buffer);
  return out;
}

inline std::ostream& error_stream() {
  if (multipers::backend_log_policy::backend_log_enabled(
          multipers::backend_log_policy::backend_log_bit::function_delaunay)) {
    return std::cerr;
  }
  static null_streambuf buffer;
  static std::ostream out(&buffer);
  return out;
}

}  // namespace function_delaunay

#define FUNCTION_DELAUNAY_COUT ::function_delaunay::log_stream()
#define FUNCTION_DELAUNAY_CERR ::function_delaunay::error_stream()
"""


def _replace_once(text: str, old: str, new: str, rel_path: Path) -> str:
    if old not in text:
        raise ValueError(f"Expected snippet not found in {rel_path.as_posix()}")
    return text.replace(old, new, 1)


def _replace_streams_active(text: str) -> str:
    """Replace std::cout/cerr only on lines where they are not commented out."""
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("//"):
            out.append(line)
            continue
        line = re.sub(r"\bstd::cout\b", "FUNCTION_DELAUNAY_COUT", line)
        line = re.sub(r"\bstd::cerr\b", "FUNCTION_DELAUNAY_CERR", line)
        out.append(line)
    return "".join(out)


def _ensure_function_delaunay_log_include(text: str, rel_path: Path) -> str:
    include = "#include <function_delaunay/log_utils.h>"
    if rel_path.name == "log_utils.h" or include in text:
        return text
    return _replace_once(text, "#pragma once\n", f"#pragma once\n\n{include}\n", rel_path)


def _patch_verbose_runtime_flag(text: str, rel_path: Path, *, ns: str, bit: str, default: str = "false", decls_before: str = "") -> str:
    """Replace `bool verbose = <default>;` with `runtime_flag<bit> verbose;` in namespace <ns>."""
    return _replace_once(
        text,
        rf"""#include<phat/helpers/misc.h>

namespace {ns} {{

  bool verbose = {default};
""",
        rf"""#include<phat/helpers/misc.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace {ns} {{

{decls_before}
#if __has_include("ext_interface/backend_log_flag.hpp")
  inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::{bit}> verbose;
#else
  bool verbose = {default};
#endif
""",
        rel_path,
    )


def _patch_multi_chunk_basic(text: str, rel_path: Path, *, bit: str, default: str) -> str:
    return _replace_once(
        text,
        rf"""#include<mpp_utils/basic.h>

namespace multi_chunk {{

  typedef mpp_utils::index index;

  bool verbose = {default};
}}
""",
        rf"""#include<mpp_utils/basic.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace multi_chunk {{

  typedef mpp_utils::index index;

#if __has_include("ext_interface/backend_log_flag.hpp")
  inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::{bit}> verbose;
#else
  bool verbose = {default};
#endif
}}
""",
        rel_path,
    )


def _patch_multi_critical_basic(text: str, rel_path: Path) -> str:
    return _replace_once(
        text,
        r"""#include<mpp_utils/basic.h>

namespace multi_critical {

    typedef mpp_utils::index index;

    bool verbose = false;
    bool very_verbose=false;
""",
        r"""#include<mpp_utils/basic.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace multi_critical {

    typedef mpp_utils::index index;

#if __has_include("ext_interface/backend_log_flag.hpp")
    inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::multi_critical> verbose;
    inline multipers::backend_log_policy::constant_flag<false> very_verbose;
#else
    bool verbose = false;
    bool very_verbose=false;
#endif
""",
        rel_path,
    )


def _patch_multi_critical_free_resolution(text: str, rel_path: Path) -> str:
    text = _replace_once(
        text,
        r"""    struct Grade_sorter {

	std::vector<Output_simplex>& data;

	Grade_sorter(std::vector<Output_simplex>& data) : data(data) {}

	bool operator() (int i, int j) {
	    Grade& g1 = data[i].grade;
	    Grade& g2 = data[j].grade;
	    if(g1.x<g2.x) {
		return true;
	    }
	    if(g1.x>g2.x) {
		return false;
	    }
	    return g1.y<g2.y;
	}
    };
""",
        r"""    struct Grade_sorter {

	std::vector<Output_simplex>& data;

	Grade_sorter(std::vector<Output_simplex>& data) : data(data) {}

	bool operator() (int i, int j) {
	    Grade& g1 = data[i].grade;
	    Grade& g2 = data[j].grade;
	    if(g1.x<g2.x) {
		return true;
	    }
	    if(g1.x>g2.x) {
		return false;
	    }
	    if(g1.y<g2.y) {
		return true;
	    }
	    if(g1.y>g2.y) {
		return false;
	    }
	    return i<j;
	}
    };
""",
        rel_path,
    )
    text = _replace_once(
        text,
        r"""std::vector<std::vector<std::pair<index,index>>> relations;
	// Syzygy with relation idx as upper boundary is saved with key idx
""",
        r"""std::vector<std::vector<std::pair<index,index>>> relations;
	// path mode only creates adjacent relations, so direct lookup avoids repeated scans
	bool path_uses_only_adjacent_relations=false;
	std::vector<index> adjacent_relations;
	// Syzygy with relation idx as upper boundary is saved with key idx
""",
        rel_path,
    )
    text, count = re.subn(
        r"\s*void get_path_of_relations\(int i,int j, std::vector<index>& rels\) \{\n"
        r"\s*assert\(i<=j\);\n"
        r"\s*if\(i==j\) \{\n"
        r"\s*return;\n"
        r"\s*\}\n"
        r"\s*int next_copy=0;\n"
        r"\s*int next_rel=0;\n"
        r"\s*for\(auto pair : relations\[i\]\) \{\n"
        r"\s*if\(pair.first>j\) \{\n"
        r"\s*break;\n"
        r"\s*\}\n"
        r"\s*next_copy=pair.first;\n"
        r"\s*next_rel=pair.second;\n"
        r"\s*\}\n"
        r"\s*rels.push_back\(next_rel\);\n"
        r"\s*get_path_of_relations\(next_copy,j,rels\);\n"
        r"\s*\}\n",
        r"""
	void get_path_of_relations(int i,int j, std::vector<index>& rels) {
	    assert(i<=j);
	    if(i==j) {
		return;
	    }
	    if(path_uses_only_adjacent_relations) {
		assert(j<=adjacent_relations.size());
		for(int k=i;k<j;k++) {
		    assert(adjacent_relations[k]>=0);
		    rels.push_back(adjacent_relations[k]);
		}
		return;
	    }
	    int next_copy=0;
	    int next_rel=0;
	    for(auto pair : relations[i]) {
		if(pair.first>j) {
		    break;
		}
		next_copy=pair.first;
		next_rel=pair.second;
	    }
	    rels.push_back(next_rel);
	    get_path_of_relations(next_copy,j,rels);
	}
""",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Expected get_path_of_relations snippet not found in {rel_path.as_posix()}")
    text = _replace_once(
        text,
        r"""		int width=1;
		input_simplex.relations.resize(input_simplex.copies.size());
""",
        r"""		int width=1;
		input_simplex.path_uses_only_adjacent_relations=!use_logpath;
		input_simplex.relations.resize(input_simplex.copies.size());
		if(input_simplex.path_uses_only_adjacent_relations && input_simplex.copies.size()>1) {
		    input_simplex.adjacent_relations.assign(input_simplex.copies.size()-1,-1);
		}
""",
        rel_path,
    )
    text = _replace_once(
        text,
        r"""			output_complex[ell+1].push_back(new_output_simplex_for_relation);
			input_simplex.relations[j-width].push_back(std::make_pair(j,idx_of_new_output_relation));
			
""",
        r"""			output_complex[ell+1].push_back(new_output_simplex_for_relation);
			input_simplex.relations[j-width].push_back(std::make_pair(j,idx_of_new_output_relation));
			if(input_simplex.path_uses_only_adjacent_relations) {
			    assert(width==1);
			    input_simplex.adjacent_relations[j-width]=idx_of_new_output_relation;
			}
			
""",
        rel_path,
    )
    text = _replace_once(
        text,
        "  };\n"
        "  \n"
        "    \n"
        "    void debug_print_output_complex(std::vector<std::vector<Output_simplex>>& complex) {\n",
        r"""  };

  template<typename Matrix_Grade>
  class Window_output_complex_accessor {

  public:

      typedef Matrix_Grade Grade;

      typedef std::vector<std::vector<Output_simplex>> Output_complex;

      Output_complex& output_complex;

      std::vector<std::vector<index>>& permutations;
      std::vector<std::vector<index>>& inv_permutations;
      int first_level;
      int matrix_count;

      Window_output_complex_accessor(Output_complex& output_complex,
				     std::vector<std::vector<index>>& permutations,
				     std::vector<std::vector<index>>& inv_permutations,
				     int first_level,
				     int matrix_count)
	  : output_complex(output_complex), permutations(permutations), inv_permutations(inv_permutations), first_level(first_level), matrix_count(matrix_count) {
      }

      int number_of_parameters() {
	  return 2;
      }

      int number_of_matrices() {
	  return matrix_count;
      }

      int output_level(int i) {
	  return first_level - i;
      }

      int number_of_columns(int i) {
	  int ell=output_level(i);
	  if(ell<0 || ell>=output_complex.size()) {
	      return 0;
	  }
	  return output_complex[ell].size();
      }

      Grade get_grade(int i, int j) {
	  int ell=output_level(i);
	  Grade g(output_complex[ell][permutations[ell][j]].grade.x,
		  output_complex[ell][permutations[ell][j]].grade.y);
	  return g;
      }

      void get_boundary(int i, int j,std::vector<index>& result) {
	  int ell=output_level(i);
	  if(ell<=0) {
	      return;
	  }
	  std::vector<index>& bd=output_complex[ell][permutations[ell][j]].boundary;
	  result.resize(bd.size());
	  for(int k=0;k<bd.size();k++) {
	      result[k]=inv_permutations[ell-1][bd[k]];
	  }
	  std::sort(result.begin(),result.end());
      }

      void clear(int i) {
      }

  };

  struct Compressed_window_matrix {
      std::vector<Grade> grades;
      std::vector<std::vector<index>> columns;
      index num_rows=0;
  };

  inline bool same_window_grade(const Grade& g1, const Grade& g2) {
      return g1.x==g2.x && g1.y==g2.y;
  }

  inline index max_window_column_index(const std::vector<index>& col) {
      if(col.empty()) {
	  return -1;
      }
      return col.back();
  }

  inline void xor_sorted_window_columns(std::vector<index>& target, const std::vector<index>& source) {
      std::vector<index> out;
      out.resize(target.size()+source.size());
      auto out_end=std::set_symmetric_difference(target.begin(),target.end(),source.begin(),source.end(),out.begin());
      out.erase(out_end,out.end());
      target.swap(out);
  }

  inline void build_window_matrices_from_output_complex(
	  std::vector<std::vector<Output_simplex>>& output_complex,
	  std::vector<std::vector<index>>& permutations,
	  std::vector<std::vector<index>>& inv_permutations,
	  int first_level,
	  int last_level,
	  std::vector<Compressed_window_matrix>& matrices) {
      int matrix_count=first_level-last_level+1;
      matrices.clear();
      matrices.resize(matrix_count);
      for(int d=0;d<matrix_count;d++) {
	  int ell=first_level-d;
	  Compressed_window_matrix& matrix=matrices[d];
	  matrix.num_rows = ell > 0 ? output_complex[ell-1].size() : 0;
	  int n=output_complex[ell].size();
	  matrix.grades.resize(n);
	  matrix.columns.resize(n);
	  for(int j=0;j<n;j++) {
	      Output_simplex& output_simplex=output_complex[ell][permutations[ell][j]];
	      matrix.grades[j]=output_simplex.grade;
	      if(ell<=0) {
		  continue;
	      }
	      std::vector<index>& bd=output_simplex.boundary;
	      std::vector<index>& col=matrix.columns[j];
	      col.resize(bd.size());
	      for(int k=0;k<bd.size();k++) {
		  col[k]=inv_permutations[ell-1][bd[k]];
	      }
	      std::sort(col.begin(),col.end());
	  }
      }
  }

  inline bool compressed_window_column_is_local(std::vector<Compressed_window_matrix>& matrices, int d, index i) {
      index p=max_window_column_index(matrices[d].columns[i]);
      return p!=-1 && same_window_grade(matrices[d].grades[i],matrices[d+1].grades[p]);
  }

  inline void compress_window_before_matrix_materialization(std::vector<Compressed_window_matrix>& matrices) {
      if(matrices.size()<2) {
	  return;
      }
      index max_d=matrices.size()-2;
      std::vector<std::vector<index>> pivots(matrices.size());
      std::vector<std::vector<char>> status(matrices.size());
      std::vector<std::vector<index>> global_index(matrices.size());
      for(int d=0;d<matrices.size();d++) {
	  pivots[d].assign(matrices[d].num_rows,-1);
	  status[d].assign(matrices[d].columns.size(),0);
	  global_index[d].assign(matrices[d].columns.size(),-1);
      }

      for(index d=0;d<=max_d;d++) {
	  index gl_index=0;
	  for(index i=0;i<matrices[d].columns.size();i++) {
	      if(matrices[d].columns[i].empty()) {
		  if(status[d][i]==0) {
		      status[d][i]=2;
		      global_index[d][i]=gl_index++;
		  }
		  continue;
	      }
	      index k=max_window_column_index(matrices[d].columns[i]);
	      index l=pivots[d][k];
	      while(compressed_window_column_is_local(matrices,d,i) && l!=-1) {
		  xor_sorted_window_columns(matrices[d].columns[i],matrices[d].columns[l]);
		  if(matrices[d].columns[i].empty()) {
		      break;
		  }
		  k=max_window_column_index(matrices[d].columns[i]);
		  l=pivots[d][k];
	      }
	      if(compressed_window_column_is_local(matrices,d,i)) {
		  k=max_window_column_index(matrices[d].columns[i]);
		  pivots[d][k]=i;
		  status[d][i]=-1;
		  status[d+1][k]=1;
		  matrices[d+1].columns[k].clear();
	      } else {
		  if(status[d][i]==0) {
		      status[d][i]=2;
		      global_index[d][i]=gl_index++;
		  }
	      }
	  }
      }

      {
	  index last=matrices.size()-1;
	  index gl_index=0;
	  for(index i=0;i<matrices[last].columns.size();i++) {
	      if(status[last][i]==0) {
		  status[last][i]=2;
	      }
	      if(status[last][i]==2) {
		  global_index[last][i]=gl_index++;
	      }
	  }
      }

      for(index d=max_d;d>=0;d--) {
	  for(index i=0;i<matrices[d].columns.size();i++) {
	      if(status[d][i]!=2) {
		  continue;
	      }
	      std::vector<index> col;
	      while(!matrices[d].columns[i].empty()) {
		  index p=max_window_column_index(matrices[d].columns[i]);
		  index j=pivots[d][p];
		  if(j!=-1) {
		      assert(status[d][j]==-1);
		      xor_sorted_window_columns(matrices[d].columns[i],matrices[d].columns[j]);
		  } else {
		      assert(status[d+1][p]==-1 || status[d+1][p]==2);
		      if(status[d+1][p]==2) {
			  col.push_back(global_index[d+1][p]);
		      }
		      matrices[d].columns[i].pop_back();
		  }
	      }
	      std::reverse(col.begin(),col.end());
	      matrices[d].columns[i].swap(col);
	  }
	  if(d==0) {
	      break;
	  }
      }

      for(index d=matrices.size()-1;d>=0;d--) {
	  Compressed_window_matrix reduced;
	  reduced.num_rows = d==matrices.size()-1 ? matrices[d].num_rows : matrices[d+1].grades.size();
	  for(index i=0;i<matrices[d].columns.size();i++) {
	      if(status[d][i]==2) {
		  reduced.grades.push_back(matrices[d].grades[i]);
		  reduced.columns.push_back(std::move(matrices[d].columns[i]));
	      }
	  }
	  matrices[d]=std::move(reduced);
	  if(d==0) {
	      break;
	  }
      }
  }

  template<typename Matrix_Grade>
  class Compressed_window_accessor {

  public:

      typedef Matrix_Grade Grade;

      std::vector<Compressed_window_matrix>& matrices;

      Compressed_window_accessor(std::vector<Compressed_window_matrix>& matrices) : matrices(matrices) {}

      int number_of_parameters() {
	  return 2;
      }

      int number_of_matrices() {
	  return matrices.size();
      }

      int number_of_columns(int i) {
	  return matrices[i].grades.size();
      }

      Grade get_grade(int i, int j) {
	  Grade g(matrices[i].grades[j].x,matrices[i].grades[j].y);
	  return g;
      }

      void get_boundary(int i, int j,std::vector<index>& result) {
	  result=std::move(matrices[i].columns[j]);
      }

      void clear(int i) {
      }

  };


    void debug_print_output_complex(std::vector<std::vector<Output_simplex>>& complex) {
""",
        rel_path,
    )
    text = _replace_once(
        text,
        r"""	void free_resolution(ParserType& parser,
			     std::vector<GradedMatrix>& result,
			     bool use_logpath=true) {
""",
        r"""	void free_resolution(ParserType& parser,
			     std::vector<GradedMatrix>& result,
			     bool use_logpath=true,
			     int target_degree=-1,
			     int target_window_radius=0,
			     bool target_prelocal=false) {
""",
        rel_path,
    )
    return _replace_once(
        text,
        r"""	std::vector<std::vector<index>> permutations, inv_permutations;
	permutations.resize(output_complex.size());
	inv_permutations.resize(output_complex.size());
	for(int i=0;i<output_complex.size();i++) {
	    long m = output_complex[i].size();
	    permutations[i].resize(m);
	    inv_permutations[i].resize(m);
	    for(int j=0;j<m;j++) {
		permutations[i][j]=j;
	    }
#if 1
	    Grade_sorter sorter(output_complex[i]);
	    std::stable_sort(permutations[i].begin(),permutations[i].end(),sorter);
#endif
	    for(int j=0;j<m;j++) {
		inv_permutations[i][permutations[i][j]]=j;
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::sort_by_grades_timer.stop();
#endif
"""
        "\t\n"
        "\t\n"
        r"""#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_graded_matrices_timer.start();
#endif
	if(multi_critical::verbose) std::cout << "Convert to graded matrices" << std::endl;
	typedef typename GradedMatrix::Grade Matrix_Grade;
	Output_complex_accessor<Matrix_Grade> accessor(output_complex,permutations,inv_permutations);
	mpp_utils::create_graded_matrices_from_pre_column_struct(accessor,result,0,false);
""",
        r"""	std::vector<std::vector<index>> permutations, inv_permutations;
	permutations.resize(output_complex.size());
	inv_permutations.resize(output_complex.size());
	auto initialize_level_order = [&](int i) {
	    if(i<0 || i>=output_complex.size()) {
		return;
	    }
	    long m = output_complex[i].size();
	    permutations[i].resize(m);
	    inv_permutations[i].resize(m);
	    for(int j=0;j<m;j++) {
		permutations[i][j]=j;
	    }
#if 1
	    Grade_sorter sorter(output_complex[i]);
	    std::sort(permutations[i].begin(),permutations[i].end(),sorter);
#endif
	    for(int j=0;j<m;j++) {
		inv_permutations[i][permutations[i][j]]=j;
	    }
	};
	int first_level=-1;
	int last_level=-1;
	if(target_degree>=0) {
	    int max_level=output_complex.size()-1;
	    if(target_degree+1>max_level) {
		result.clear();
		return;
	    }
	    int upper_extra=std::min(target_window_radius,max_level-(target_degree+1));
	    int lower_extra=std::min(target_window_radius,target_degree);
	    first_level=target_degree+1+upper_extra;
	    last_level=target_degree-lower_extra;
	    for(int ell=first_level;ell>=last_level;ell--) {
		initialize_level_order(ell);
	    }
	    if(last_level>0) {
		initialize_level_order(last_level-1);
	    }
	} else {
	    for(int i=0;i<output_complex.size();i++) {
		initialize_level_order(i);
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::sort_by_grades_timer.stop();
#endif


#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_graded_matrices_timer.start();
#endif
	if(multi_critical::verbose) std::cout << "Convert to graded matrices" << std::endl;
	typedef typename GradedMatrix::Grade Matrix_Grade;
	if(target_degree>=0) {
	    int matrix_count=first_level-last_level+1;
	    int number_of_rows_in_last_matrix = last_level > 0 ? output_complex[last_level-1].size() : 0;
	    if(target_prelocal) {
		std::vector<Compressed_window_matrix> compressed_matrices;
		build_window_matrices_from_output_complex(output_complex,permutations,inv_permutations,first_level,last_level,compressed_matrices);
		compress_window_before_matrix_materialization(compressed_matrices);
		number_of_rows_in_last_matrix = compressed_matrices.empty() ? 0 : compressed_matrices.back().num_rows;
		Compressed_window_accessor<Matrix_Grade> accessor(compressed_matrices);
		mpp_utils::create_graded_matrices_from_pre_column_struct(accessor,result,number_of_rows_in_last_matrix,false);
	    } else {
		Window_output_complex_accessor<Matrix_Grade> accessor(output_complex,permutations,inv_permutations,first_level,matrix_count);
		mpp_utils::create_graded_matrices_from_pre_column_struct(accessor,result,number_of_rows_in_last_matrix,false);
	    }
	} else {
	    Output_complex_accessor<Matrix_Grade> accessor(output_complex,permutations,inv_permutations);
	    mpp_utils::create_graded_matrices_from_pre_column_struct(accessor,result,0,false);
	}
""",
        rel_path,
    )


def _patch_multi_critical_graded_matrix(text: str, rel_path: Path) -> str:
    text = _replace_once(
        text,
        r"""M1.row_grades.clear();
	    for(index j=0;j<M1.num_rows;j++) {
		M1.row_grades.push_back(M2.grades[j]);
	    }
""",
        r"""M1.row_grades.assign(M2.grades.begin(),M2.grades.begin()+M1.num_rows);
""",
        rel_path,
    )
    return _replace_once(
        text,
        r"""	M1.row_grades.clear();
	for(int i=0;i<n2;i++) {
	    M1.row_grades.push_back(M2.grades[i]);
	}
""",
        r"""	M1.row_grades.assign(M2.grades.begin(),M2.grades.begin()+n2);
""",
        rel_path,
    )


def _patch_multi_critical_pre_column_struct(text: str, rel_path: Path) -> str:
    text = _replace_once(text, "#include<cstring>\n", "#include<cstring>\n#include<utility>\n", rel_path)
    text, count = re.subn(
        r"(\s*void get_boundary\(int i, int j,std::vector<index>& result\) \{\n)"
        r"\s*result=pre_columns\[i\]\[j\]\.boundary;\n"
        r"(\s*\}\n)",
        r"\1\t  result=std::move(pre_columns[i][j].boundary);\n\2",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Expected pre-column get_boundary snippet not found in {rel_path.as_posix()}")
    return _replace_once(
        text,
        r"""matrix.set_col(j,bd);
""",
        r"""matrix.set_col(j,std::move(bd));
""",
        rel_path,
    )


def _patch_scc_basic(text: str, rel_path: Path, *, bit: str) -> str:
    return _replace_once(
        text,
        r"""#include <stdint.h>

namespace scc {

  typedef int64_t index;

  bool verbose = false;
""",
        rf"""#include <stdint.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace scc {{

  typedef int64_t index;

#if __has_include("ext_interface/backend_log_flag.hpp")
  inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::{bit}> verbose;
#else
  bool verbose = false;
#endif
""",
        rel_path,
    )


def _patch_deg_rips_edge_copy_reducer(text: str, rel_path: Path) -> str:
    text, count = re.subn(
        r"(?m)^([ \t]*test_timer_\d+\.(?:resume|stop)\(\);)[ \t]*$",
        r"#if DEG_RIPS_TIMERS\n\1\n#endif",
        text,
    )
    if count != 4:
        raise ValueError(
            f"Expected 4 deg_rips test timer calls in {rel_path.as_posix()}, found {count}"
        )

    text = _replace_once(
        text,
        """\tlong count_deleted=0;
\tstd::vector<Edge_copy> flattened_edge_copies;
""",
        """\tlong count_deleted=0;
\tstd::vector<Edge_copy> flattened_edge_copies;
\tstd::vector<int> potential_dominators;
\tstd::vector<int> common_neighbors;
\tstd::vector<Grade> staircase_av,staircase_bv,staircase;
\tpotential_dominators.reserve(n);
\tcommon_neighbors.reserve(n);
\tstaircase_av.reserve(16);
\tstaircase_bv.reserve(16);
\tstaircase.reserve(16);
""",
        rel_path,
    )
    text = _replace_once(
        text,
        """\t    std::vector<int> potential_dominators;
\t    std::vector<int> common_neighbors;
""",
        """\t    potential_dominators.clear();
\t    common_neighbors.clear();
""",
        rel_path,
    )
    return _replace_once(
        text,
        """\t\tstd::vector<Grade> staircase_av,staircase_bv,staircase;
""",
        """\t\tstaircase_av.clear();
\t\tstaircase_bv.clear();
\t\tstaircase.clear();
""",
        rel_path,
    )


def _build_explicit_targets(targets: dict[Path, Callable[[str, Path], str]]) -> dict[Path, str]:
    out: dict[Path, str] = {}
    for rel_path, patch_fn in sorted(targets.items()):
        original = (REPO_ROOT / rel_path).read_text()
        patched = patch_fn(original, rel_path)
        if patched != original:
            out[rel_path] = patched
    return out


def _function_delaunay_targets() -> dict[Path, str]:
    base = REPO_ROOT / "ext" / "function_delaunay" / "include" / "function_delaunay"
    out: dict[Path, str] = {}
    for path in sorted(base.glob("*.h")):
        if path.name == "log_utils.h":
            continue
        rel = path.relative_to(REPO_ROOT)
        original = path.read_text()
        patched = _replace_streams_active(original)
        if patched != original:
            patched = _ensure_function_delaunay_log_include(patched, rel)
            out[rel] = patched
    out[Path("ext/function_delaunay/include/function_delaunay/log_utils.h")] = FUNCTION_DELAUNAY_LOG_UTILS
    out.update(
        _build_explicit_targets(
            {
                Path("ext/function_delaunay/mpfree_mod/include/mpfree/global.h"): (
                    lambda text, rel: _patch_verbose_runtime_flag(text, rel, ns="mpfree", bit="function_delaunay")
                ),
                Path("ext/function_delaunay/multi_chunk_mod/include/multi_chunk/basic.h"): (
                    lambda text, rel: _patch_multi_chunk_basic(
                        text,
                        rel,
                        bit="function_delaunay",
                        default="true",
                    )
                ),
            }
        )
    )
    return out


def _mpfree_targets() -> dict[Path, str]:
    return _build_explicit_targets(
        {
            Path("ext/mpfree/include/mpfree/global.h"): (
                lambda text, rel: _patch_verbose_runtime_flag(text, rel, ns="mpfree", bit="mpfree")
            ),
        }
    )


def _multi_critical_logs_targets() -> dict[Path, str]:
    return _build_explicit_targets(
        {
            Path("ext/multi_critical/include/multi_critical/basic.h"): _patch_multi_critical_basic,
            Path("ext/multi_critical/mpfree_mod/include/mpfree/global.h"): (
                lambda text, rel: _patch_verbose_runtime_flag(text, rel, ns="mpfree", bit="multi_critical")
            ),
            Path("ext/multi_critical/scc_mod/include/scc/basic.h"): (
                lambda text, rel: _patch_scc_basic(text, rel, bit="multi_critical")
            ),
        }
    )


def _multi_critical_features_targets() -> dict[Path, str]:
    return _build_explicit_targets(
        {
            Path("ext/multi_critical/include/multi_critical/free_resolution.h"): _patch_multi_critical_free_resolution,
            Path("ext/multi_critical/mpp_utils_mod/include/mpp_utils/Graded_matrix.h"): _patch_multi_critical_graded_matrix,
            Path("ext/multi_critical/mpp_utils_mod/include/mpp_utils/create_graded_matrices_from_pre_column_struct.h"): _patch_multi_critical_pre_column_struct,
        }
    )


def _deg_rips_targets() -> dict[Path, str]:
    return _build_explicit_targets(
        {
            Path("ext/deg_rips/include/deg_rips/Edge_domination_checker.h"): _patch_deg_rips_edge_copy_reducer,
        }
    )


GENERATORS = {
    "deg_rips": (
        _deg_rips_targets,
        PATCH_DIR / "deg_rips_edge_copy_reducer.patch",
    ),
    "function_delaunay": (
        _function_delaunay_targets,
        PATCH_DIR / "function_delaunay_runtime_logs.patch",
    ),
    "mpfree": (
        _mpfree_targets,
        PATCH_DIR / "mpfree_runtime_logs.patch",
    ),
    "multi_critical_logs": (
        _multi_critical_logs_targets,
        PATCH_DIR / "multi_critical_runtime_logs.patch",
    ),
    "multi_critical_features": (
        _multi_critical_features_targets,
        PATCH_DIR / "multi_critical_features.patch",
    ),
}


def _build_patch(library: str) -> str:
    target_fn, _ = GENERATORS[library]
    outputs = []
    for rel_path, patched in target_fn().items():
        original_path = REPO_ROOT / rel_path
        if original_path.exists():
            original = original_path.read_text().splitlines(keepends=True)
            fromfile = f"a/{rel_path.as_posix()}"
        else:
            original = []
            fromfile = "/dev/null"
        updated = patched.splitlines(keepends=True)
        outputs.extend(
            difflib.unified_diff(
                original,
                updated,
                fromfile=fromfile,
                tofile=f"b/{rel_path.as_posix()}",
            )
        )
    return "".join(outputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate vendored patch files.")
    parser.add_argument("library", choices=sorted(GENERATORS))
    parser.add_argument("--output", type=Path, default=None, help="Patch output path")
    args = parser.parse_args(argv)

    patch_text = _build_patch(args.library)
    if not patch_text:
        print(f"No changes generated for {args.library}.", file=sys.stderr)
        return 1

    _, default_output = GENERATORS[args.library]
    output = default_output if args.output is None else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patch_text)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
