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


def _replace_streams(text: str) -> str:
    text = re.sub(r"\bstd::cout\b", "FUNCTION_DELAUNAY_COUT", text)
    text = re.sub(r"\bstd::cerr\b", "FUNCTION_DELAUNAY_CERR", text)
    return text


def _ensure_function_delaunay_log_include(text: str, rel_path: Path) -> str:
    include = "#include <function_delaunay/log_utils.h>"
    if rel_path.name == "log_utils.h" or include in text:
        return text
    return _replace_once(text, "#pragma once\n", f"#pragma once\n\n{include}\n", rel_path)


def _patch_mpfree_global(text: str, rel_path: Path) -> str:
    return _replace_once(
        text,
        r"""#include<phat/helpers/misc.h>

namespace mpfree {

  bool verbose = false;
""",
        r"""#include<phat/helpers/misc.h>

#if __has_include("ext_interface/backend_log_flag.hpp")
#include "ext_interface/backend_log_flag.hpp"
#endif

namespace mpfree {

#if __has_include("ext_interface/backend_log_flag.hpp")
  inline multipers::backend_log_policy::runtime_flag<multipers::backend_log_policy::backend_log_bit::mpfree> verbose;
#else
  bool verbose = false;
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
    return _replace_once(
        text,
        r"""#if 1
	    Grade_sorter sorter(output_complex[i]);
	    std::stable_sort(permutations[i].begin(),permutations[i].end(),sorter);
#endif
""",
        r"""#if 1
	    Grade_sorter sorter(output_complex[i]);
	    std::sort(permutations[i].begin(),permutations[i].end(),sorter);
#endif
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
        patched = _replace_streams(original)
        if patched != original:
            patched = _ensure_function_delaunay_log_include(patched, rel)
            out[rel] = patched
    out[Path("ext/function_delaunay/include/function_delaunay/log_utils.h")] = FUNCTION_DELAUNAY_LOG_UTILS
    out.update(
        _build_explicit_targets(
            {
                Path("ext/function_delaunay/mpfree_mod/include/mpfree/global.h"): _patch_mpfree_global,
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
            Path("ext/mpfree/include/mpfree/global.h"): _patch_mpfree_global,
        }
    )


def _multi_critical_targets() -> dict[Path, str]:
    return _build_explicit_targets(
        {
            Path("ext/multi_critical/include/multi_critical/basic.h"): _patch_multi_critical_basic,
            Path("ext/multi_critical/include/multi_critical/free_resolution.h"): _patch_multi_critical_free_resolution,
            Path("ext/multi_critical/mpfree_mod/include/mpfree/global.h"): _patch_mpfree_global,
            Path("ext/multi_critical/mpp_utils_mod/include/mpp_utils/Graded_matrix.h"): _patch_multi_critical_graded_matrix,
            Path("ext/multi_critical/mpp_utils_mod/include/mpp_utils/create_graded_matrices_from_pre_column_struct.h"): _patch_multi_critical_pre_column_struct,
            Path("ext/multi_critical/multi_chunk_mod/include/multi_chunk/basic.h"): (
                lambda text, rel: _patch_multi_chunk_basic(
                    text,
                    rel,
                    bit="multi_critical",
                    default="true",
                )
            ),
            Path("ext/multi_critical/scc_mod/include/scc/basic.h"): (
                lambda text, rel: _patch_scc_basic(text, rel, bit="multi_critical")
            ),
        }
    )


GENERATORS = {
    "function_delaunay": (
        _function_delaunay_targets,
        PATCH_DIR / "function_delaunay_runtime_logs.patch",
    ),
    "mpfree": (
        _mpfree_targets,
        PATCH_DIR / "mpfree_runtime_logs.patch",
    ),
    "multi_critical": (
        _multi_critical_targets,
        PATCH_DIR / "multi_critical_runtime_logs.patch",
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
    parser = argparse.ArgumentParser(description="Generate vendored log-control patch files.")
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
