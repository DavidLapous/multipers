#ifndef H0V_MERGE_TREE_H
#define H0V_MERGE_TREE_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/functional/hash.hpp>

struct Bar {
  Bar() : dim(-1), birth(-1), death(-1) {}

  Bar(int dim, int birth, int death) : dim(dim), birth(birth), death(death) {}

  int dim;
  int birth;
  int death;
};

class Naive_bottleneck_forest {
 public:
  using index = int;

  Naive_bottleneck_forest() {}

  // assumes that ids of vertices are continuous and start with 0
  Naive_bottleneck_forest(unsigned int numberOfVertices) : forest_(numberOfVertices), weights_() {}

  friend void swap(Naive_bottleneck_forest &mf1, Naive_bottleneck_forest &mf2) {
    mf1.forest_.swap(mf2.forest_);
    mf1.weights_.swap(mf2.weights_);
  }

  void add_edge(index firstVertexIndex, index secondVertexIndex, index edgeWeight) {
    forest_[firstVertexIndex].children.insert(secondVertexIndex);
    forest_[secondVertexIndex].children.insert(firstVertexIndex);
    weights_.emplace(_get_edge(firstVertexIndex, secondVertexIndex), edgeWeight);
  }

  void swap_out_edges(index inVertex1, index inVertex2, index outVertex1, index outVertex2, index outWeight) {
    if (forest_[inVertex1].parent == inVertex2) {
      forest_[inVertex1].parent = -1;
      forest_[inVertex2].children.erase(inVertex1);
    } else {
      forest_[inVertex2].parent = -1;
      forest_[inVertex1].children.erase(inVertex2);
    }
    weights_.erase(_get_edge(inVertex1, inVertex2));

    if (forest_[outVertex1].parent == -1) {
      forest_[outVertex1].parent = outVertex2;
      forest_[outVertex2].children.insert(outVertex1);
    } else {
      if (forest_[outVertex2].parent != -1) _reroot(outVertex2);
      forest_[outVertex2].parent = outVertex1;
      forest_[outVertex1].children.insert(outVertex2);
    }
    weights_.emplace(_get_edge(outVertex1, outVertex2), outWeight);
  }

  // to call once the tree finished
  void root() {
    for (unsigned int i = 0; i < forest_.size(); ++i) {
      if (forest_[i].parent == -1) {
        _root(i);
      }
    }
  }

  // assumes root() was already called and that both vertices are in the same
  // connected component
  index get_bootleneck_weight(index id1, index id2) const {
    auto get_weight = [&](index id) {
      return forest_[id].parent == -1 ? -1 : weights_.at(_get_edge(id, forest_[id].parent));
    };

    std::unordered_map<index, index> partialBottleneck;
    index r = id1;
    index w = 0;
    partialBottleneck.emplace(r, w);
    while (forest_[r].parent != -1) {
      w = std::max(get_weight(r), w);
      partialBottleneck.emplace(forest_[r].parent, w);
      r = forest_[r].parent;
    }
    r = id2;
    auto bIt = partialBottleneck.find(r);
    w = 0;
    while (forest_[r].parent != -1 && bIt == partialBottleneck.end()) {
      w = std::max(get_weight(r), w);
      r = forest_[r].parent;
      bIt = partialBottleneck.find(r);
    }
    return std::max(w, bIt->second);
  }

  void update_weight(index id1, index id2, index newWeight) { weights_.at(_get_edge(id1, id2)) = newWeight; }

  void print(std::ostream &stream) const {
    stream << "MSF:\n";
    unsigned int c = 0;
    for (const Node &n : forest_) {
      stream << "[" << c << "] parent: ";
      stream << n.parent << "\n";
      stream << "[" << c << "] children: ";
      for (index id : n.children) stream << id << " ";
      stream << "\n";
      if (n.parent != -1) {
        stream << "[" << c << "] weight of parent edge: ";
        stream << weights_.at(_get_edge(c, n.parent)) << "\n";
      }
      ++c;
    }
  }

 private:
  struct Node {
    index parent = -1;
    std::set<index> children = {};
  };

  std::vector<Node> forest_;
  std::unordered_map<std::pair<index, index>, index, boost::hash<std::pair<index, index>>>
      weights_;  // TODO: test other map types and having a pair as key is ugly

  void _root(index id) {
    const Node &p = forest_[id];
    for (index chID : p.children) {
      Node &ch = forest_[chID];
      ch.children.erase(id);
      ch.parent = id;
      _root(chID);
    }
  }

  void _reroot(index id) {
    index newParent = -1;
    index current = id;
    index oldParent = forest_[id].parent;
    forest_[current].parent = newParent;
    forest_[current].children.insert(oldParent);
    newParent = current;
    current = oldParent;
    oldParent = forest_[current].parent;
    while (oldParent != -1) {
      forest_[current].parent = newParent;
      forest_[current].children.erase(newParent);
      forest_[current].children.insert(oldParent);
      newParent = current;
      current = oldParent;
      oldParent = forest_[current].parent;
    }
    forest_[current].parent = newParent;
    forest_[current].children.erase(newParent);
  }

  std::pair<index, index> _get_edge(index id1, index id2) const {
    return std::make_pair(std::min(id1, id2), std::max(id1, id2));
  }
};

class Naive_merge_forest {
 public:
  using index = int;

  Naive_merge_forest() {}

  Naive_merge_forest(unsigned int numberOfSimplices, unsigned int numberOfVertices)
      : forest_(numberOfVertices * 2 - 1),
        barcode_(numberOfSimplices, numberOfVertices),
        positionToID_(numberOfSimplices, -1),
        nextVertexIndex_(0),
        nextEdgeIndex_(numberOfVertices),
        minimumSpanningForest_(numberOfVertices) {}

  friend void swap(Naive_merge_forest &mf1, Naive_merge_forest &mf2) {
    mf1.forest_.swap(mf2.forest_);
    swap(mf1.barcode_, mf2.barcode_);
    mf1.positionToID_.swap(mf2.positionToID_);
    std::swap(mf1.nextVertexIndex_, mf2.nextVertexIndex_);
    std::swap(mf1.nextEdgeIndex_, mf2.nextEdgeIndex_);
    swap(mf1.minimumSpanningForest_, mf2.minimumSpanningForest_);
  }

  // assumes that merge tree is constructed fully with add_edge before any swap
  // is done with vine
  bool add_edge(index edgeIndex, index firstVertexIndex, index secondVertexIndex) {
    index r1 = _get_root(positionToID_[firstVertexIndex]);
    index r2 = _get_root(positionToID_[secondVertexIndex]);

    if (r1 == r2) {
      barcode_.add_positive_edge(edgeIndex);
      return false;  // positive edge
    }

    assert(positionToID_[edgeIndex] == -1 && "Edge was already added");
    positionToID_[edgeIndex] = nextEdgeIndex_;

    forest_[r1].parent = nextEdgeIndex_;
    forest_[r2].parent = nextEdgeIndex_;
    forest_[nextEdgeIndex_].leftChild = r1;
    forest_[nextEdgeIndex_].rightChild = r2;
    index rep1 = _is_vertex(r1) ? r1 : forest_[r1].representative;
    index rep2 = _is_vertex(r2) ? r2 : forest_[r2].representative;
    forest_[nextEdgeIndex_].representative = rep1 < rep2 ? rep1 : rep2;
    index pair =
        (forest_[nextEdgeIndex_].representative == rep1) ? forest_[rep2].representative : forest_[rep1].representative;
    barcode_.add_negative_edge(pair, edgeIndex);

    ++nextEdgeIndex_;

    minimumSpanningForest_.add_edge(positionToID_[firstVertexIndex], positionToID_[secondVertexIndex], edgeIndex);

    return true;  // negative edge
  }

  void add_vertex(index vertexIndex) {
    positionToID_[vertexIndex] = nextVertexIndex_;
    forest_[nextVertexIndex_++].representative = vertexIndex;
    barcode_.add_vertex(vertexIndex);
  }

  void initialize() {
    barcode_.barcode_.resize(barcode_.nextBarIndex_);
    minimumSpanningForest_.root();
  }

  void vertex_swap(index position) {
    index nca = _get_nearest_common_ancestor(positionToID_[position], positionToID_[position + 1]);

    if (nca == -1) {  // not in the same connected component
      barcode_.birth_birth_swap(position);
    } else {
      index rightRep = _is_vertex(forest_[nca].rightChild) ? forest_[nca].rightChild
                                                           : forest_[forest_[nca].rightChild].representative;
      index leftRep =
          _is_vertex(forest_[nca].leftChild) ? forest_[nca].leftChild : forest_[forest_[nca].leftChild].representative;
      index ncaPairID = (forest_[nca].representative == rightRep) ? leftRep : rightRep;
      index ncaPos = barcode_.death(forest_[ncaPairID].representative);

      index e1 = barcode_.death(position);
      index e2 = barcode_.death(position + 1);

      if ((e1 != -1 && e1 < ncaPos) || (e2 != -1 && e2 < ncaPos)) {
        barcode_.birth_birth_swap(position);
      }
    }

    _update_representative(nca, positionToID_[position], positionToID_[position + 1]);
    std::swap(forest_[positionToID_[position]].representative, forest_[positionToID_[position + 1]].representative);
    std::swap(positionToID_[position], positionToID_[position + 1]);
  }

  void vertex_edge_swap(index position, index eV1Position, index eV2Position) {
    if (_is_positive_edge(position + 1)) {
      barcode_.birth_birth_swap(position);
    } else {
      barcode_.birth_death_swap(position);
      minimumSpanningForest_.update_weight(positionToID_[eV1Position], positionToID_[eV2Position], position);
    }
    forest_[positionToID_[position]].representative = position + 1;
    std::swap(positionToID_[position], positionToID_[position + 1]);
  }

  void edge_vertex_swap(index position, index eV1Position, index eV2Position) {
    if (_is_positive_edge(position)) {
      barcode_.birth_birth_swap(position);
    } else {
      barcode_.death_birth_swap(position);
      minimumSpanningForest_.update_weight(positionToID_[eV1Position], positionToID_[eV2Position], position + 1);
    }
    forest_[positionToID_[position + 1]].representative = position;
    std::swap(positionToID_[position], positionToID_[position + 1]);
  }

  void edge_edge_swap(index position, index e1V1Position, index e1V2Position, index e2V1Position, index e2V2Position) {
    if (_is_positive_edge(position)) {
      _pos_edge_edge_switch(position, e2V1Position, e2V2Position);
      return;
    }

    if (_is_positive_edge(position + 1)) {
      _neg_edge_pos_edge_switch(position, e1V1Position, e1V2Position, e2V1Position, e2V2Position);
      return;
    }

    _neg_edge_neg_edge_switch(position, e1V1Position, e1V2Position, e2V1Position, e2V2Position);
  }

  const std::vector<Bar> &get_barcode() const { return barcode_.barcode_; }

  int get_dimension(index position) { return !_is_vertex(positionToID_[position]); }

  inline friend std::ostream &operator<<(std::ostream &stream, Naive_merge_forest &structure) {
    for (unsigned int i = 0; i < structure.forest_.size(); ++i) {
      if (structure.forest_[i].parent == -1) structure._print_subtree(stream, "", i, false);
    }
    structure.minimumSpanningForest_.print(stream);
    return stream;
  }

 private:
  struct Node {
    index parent = -1;
    index leftChild = -1;
    index rightChild = -1;
    index representative = -1;  // cc rep for edges, position for vertices
  };

  struct Barcode {
    Barcode() {}

    Barcode(int numberOfSimplices, [[maybe_unused]] int numberOfVertices)
        : barcode_(numberOfSimplices), positionToBar_(numberOfSimplices), nextBarIndex_(0) {}

    friend void swap(Barcode &mf1, Barcode &mf2) {
      mf1.barcode_.swap(mf2.barcode_);
      mf1.positionToBar_.swap(mf2.positionToBar_);
      std::swap(mf1.nextBarIndex_, mf2.nextBarIndex_);
    }

    void add_birth(index birth) {
      positionToBar_[birth] = nextBarIndex_;
      barcode_[nextBarIndex_].birth = birth;
      ++nextBarIndex_;
    }

    void add_vertex(index birth) {
      barcode_[nextBarIndex_].dim = 0;
      add_birth(birth);
    }

    void add_positive_edge(index birth) {
      barcode_[nextBarIndex_].dim = 1;
      add_birth(birth);
    }

    void add_negative_edge(index birth, index death) {
      index b = positionToBar_[birth];
      positionToBar_[death] = b;
      barcode_[b].death = death;
    }

    void birth_birth_swap(index position) {
      std::swap(barcode_[positionToBar_[position]].birth, barcode_[positionToBar_[position + 1]].birth);
      std::swap(positionToBar_[position], positionToBar_[position + 1]);
    }

    void birth_death_swap(index position) {
      std::swap(barcode_[positionToBar_[position]].birth, barcode_[positionToBar_[position + 1]].death);
      std::swap(positionToBar_[position], positionToBar_[position + 1]);
    }

    void death_death_swap(index position) {
      std::swap(barcode_[positionToBar_[position]].death, barcode_[positionToBar_[position + 1]].death);
      std::swap(positionToBar_[position], positionToBar_[position + 1]);
    }

    void death_birth_swap(index position) {
      std::swap(barcode_[positionToBar_[position]].death, barcode_[positionToBar_[position + 1]].birth);
      std::swap(positionToBar_[position], positionToBar_[position + 1]);
    }

    index death(index birth) { return barcode_[positionToBar_[birth]].death; }

    std::vector<Bar> barcode_;
    std::vector<index> positionToBar_;
    index nextBarIndex_;
  };

  std::vector<Node> forest_;
  Barcode barcode_;
  std::vector<index> positionToID_;
  index nextVertexIndex_;
  index nextEdgeIndex_;
  Naive_bottleneck_forest minimumSpanningForest_;

  bool _is_vertex(index nodeIndex) const {
    const Node &n = forest_[nodeIndex];
    return n.leftChild == -1 && n.rightChild == -1;
  }

  bool _is_positive_edge(index position) const { return positionToID_[position] == -1; }

  index _get_root(index node) const {
    while (forest_[node].parent != -1) {
      node = forest_[node].parent;
    }
    return node;
  }

  index _get_nearest_common_ancestor(index firstIndex, index secondIndex) const {
    std::set<index> path;
    index r = firstIndex;
    path.insert(r);
    while (forest_[r].parent != -1) {
      r = forest_[r].parent;
      path.insert(r);
    }
    r = secondIndex;
    while (r != -1 && path.find(r) == path.end()) {
      r = forest_[r].parent;
    }
    return r;
  }

  index _is_ancestor_of(index ancID, index descID, index endID = -1) const {
    index r = descID;
    while (r != endID && forest_[r].parent != ancID) {
      r = forest_[r].parent;
    }
    return r;  // either returns endID or the child of the ancestor it came
               // through.
  }

  void _update_representative(index pos, index oldRep, index newRep) {
    index r = pos;
    while (r != -1 && forest_[r].representative == oldRep) {
      forest_[r].representative = newRep;
      r = forest_[r].parent;
    }
  }

  void _neg_edge_neg_edge_switch(index position,
                                 index e1V1Position,
                                 index e1V2Position,
                                 index e2V1Position,
                                 index e2V2Position) {
    minimumSpanningForest_.update_weight(positionToID_[e1V1Position], positionToID_[e1V2Position], position + 1);
    minimumSpanningForest_.update_weight(positionToID_[e2V1Position], positionToID_[e2V2Position], position);

    index e1 = positionToID_[position];
    index e2 = positionToID_[position + 1];

    if (forest_[e2].leftChild != e1 && forest_[e2].rightChild != e1) {
      barcode_.death_death_swap(position);
      std::swap(positionToID_[position], positionToID_[position + 1]);
      return;
    }

    index t1 = forest_[e1].leftChild;
    index t2 = forest_[e1].rightChild;
    index t3;
    if (forest_[e2].leftChild == e1)
      t3 = forest_[e2].rightChild;
    else
      t3 = forest_[e2].leftChild;

    index u = _is_vertex(t1) ? t1 : forest_[t1].representative;
    index v = _is_vertex(t2) ? t2 : forest_[t2].representative;
    index w = _is_vertex(t3) ? t3 : forest_[t3].representative;
    if (forest_[u].representative < forest_[v].representative) {
      std::swap(u, v);
      std::swap(t1, t2);
    }

    index t = _is_ancestor_of(e1, positionToID_[e2V1Position], e2);
    if (t != t1 && t != t2) t = _is_ancestor_of(e1, positionToID_[e2V2Position], e2);

    if (t == t1) {
      forest_[e1].leftChild = t1;
      forest_[e2].leftChild = t2;
      forest_[e1].representative = (forest_[u].representative < forest_[w].representative) ? u : w;
      forest_[e2].representative = (forest_[forest_[e1].representative].representative < forest_[v].representative)
                                       ? forest_[e1].representative
                                       : v;
      forest_[t2].parent = e2;
    } else {
      forest_[e1].leftChild = t2;
      forest_[e2].leftChild = t1;
      forest_[e1].representative = (forest_[v].representative < forest_[w].representative) ? v : w;
      forest_[e2].representative = (forest_[forest_[e1].representative].representative < forest_[u].representative)
                                       ? forest_[e1].representative
                                       : u;
      forest_[t1].parent = e2;
    }
    forest_[e1].rightChild = t3;
    forest_[e2].rightChild = e1;
    forest_[t3].parent = e1;

    if (t == t2 || forest_[u].representative < forest_[w].representative) {
      barcode_.death_death_swap(position);
    }
  }

  void _neg_edge_pos_edge_switch(index position,
                                 index e1V1Position,
                                 index e1V2Position,
                                 index e2V1Position,
                                 index e2V2Position) {
    index bottleneck =
        minimumSpanningForest_.get_bootleneck_weight(positionToID_[e2V1Position], positionToID_[e2V2Position]);

    if (bottleneck == position) {
      minimumSpanningForest_.swap_out_edges(positionToID_[e1V1Position],
                                            positionToID_[e1V2Position],
                                            positionToID_[e2V1Position],
                                            positionToID_[e2V2Position],
                                            position);
    } else {
      minimumSpanningForest_.update_weight(positionToID_[e1V1Position], positionToID_[e1V2Position], position + 1);
      barcode_.death_birth_swap(position);
      std::swap(positionToID_[position], positionToID_[position + 1]);
    }
  }

  void _pos_edge_edge_switch(index position, index e2V1Position, index e2V2Position) {
    if (_is_positive_edge(position + 1)) {
      barcode_.birth_birth_swap(position);
    } else {
      barcode_.birth_death_swap(position);
      minimumSpanningForest_.update_weight(positionToID_[e2V1Position], positionToID_[e2V2Position], position);
      std::swap(positionToID_[position], positionToID_[position + 1]);
    }
  }

  void _print_subtree(std::ostream &stream, const std::string &prefix, index id, bool isLeft) {
    if (id != -1) {
      stream << prefix;
      stream << (isLeft ? "├──" : "└──");

      // print the value of the node
      stream << id << std::endl;

      // enter the next tree level - left and right branch
      _print_subtree(stream, prefix + (isLeft ? "│   " : "    "), forest_[id].leftChild, true);
      _print_subtree(stream, prefix + (isLeft ? "│   " : "    "), forest_[id].rightChild, false);
    }
  }
};

#endif  // H0V_MERGE_TREE_H
