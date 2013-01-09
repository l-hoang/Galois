/** Cuthull-McKee -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/CheckedObject.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/Graph.h"
#ifdef GALOIS_EXP
#include "Galois/PriorityScheduling.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "Galois/Atomic.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>
#include <queue>
#include <numeric>

static const char* name = "Cuthill-McKee";
static const char* desc = "";
static const char* url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);
static cll::opt<bool> useSimplePseudo("useSimplePseudo",
    cll::desc("use simple pseudo-peripheral algorithm"),
    cll::init(false));


typedef unsigned int DistType;

static const DistType DIST_INFINITY = std::numeric_limits<DistType>::max() - 1;

namespace {

//****** Work Item and Node Data Definitions ******
struct SNode {
  DistType dist;
  unsigned int degree;
  unsigned int id;
  bool done;
};

//typedef Galois::Graph::LC_Linear_Graph<SNode, void> Graph;
typedef Galois::Graph::LC_CSR_Graph<SNode, void> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

std::vector<GNode> perm;

std::ostream& operator<<(std::ostream& os, const SNode& n) {
  os << "(id: " << n.id;
  os << ", dist: ";
  if (n.dist == DIST_INFINITY)
    os << "Inf";
  else
    os << n.dist;
  os << ", degree: " << n.degree << ")";
  return os;
}

struct GNodeIndexer: public std::unary_function<GNode,unsigned int> {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, Galois::MethodFlag::NONE).dist;// >> 2;
  }
};

struct sortDegFn {
  bool operator()(const GNode& lhs, const GNode& rhs) const {
    return
      std::distance(graph.edge_begin(lhs, Galois::MethodFlag::NONE),
		    graph.edge_end(lhs, Galois::MethodFlag::NONE))
      <
      std::distance(graph.edge_begin(rhs, Galois::MethodFlag::NONE),
		    graph.edge_end(rhs, Galois::MethodFlag::NONE))
      ;
  }
};

struct UnsignedIndexer: public std::unary_function<unsigned,unsigned> {
  unsigned operator()(unsigned x) const { return x;}
};

struct default_reduce {
  template<typename T>
  void operator()(T& dest, T& src) {
    T::reduce(dest,src);
  }
};

struct BFS {
  typedef std::deque<size_t> Counts;

  //! Compute BFS levels
  struct Process {
    typedef int tt_does_not_need_aborts;

    void operator()(const GNode& n, Galois::UserContext<GNode>& ctx) const {
      SNode& data = graph.getData(n, Galois::MethodFlag::NONE);
      DistType newDist = data.dist + 1;
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
             ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
        DistType oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            ctx.push(dst);
            break;
          }
        }
      }
    }
  };

  //! Compute histogram of levels
  struct CountLevels {
    Counts counts;
    bool reset;

    explicit CountLevels(bool r): reset(r) { }

    void operator()(const GNode& n) {
      SNode& data = graph.getData(n, Galois::MethodFlag::NONE);
      
      assert(data.dist != DIST_INFINITY);

      if (counts.size() <= data.dist)
        counts.resize(data.dist + 1);
      ++counts[data.dist];
      if (reset)
        data.dist = DIST_INFINITY;
    }

    static void reduce(CountLevels& dest, CountLevels& src) {
      if (dest.counts.size() < src.counts.size())
        dest.counts.resize(src.counts.size());
      std::transform(src.counts.begin(), src.counts.end(), dest.counts.begin(), dest.counts.begin(), std::plus<size_t>());
    }
  };

  static void initNode(const GNode& n) {
    SNode& data = graph.getData(n, Galois::MethodFlag::NONE);
    data.degree = std::distance(graph.edge_begin(n, Galois::MethodFlag::NONE),
        graph.edge_end(n, Galois::MethodFlag::NONE));
    resetNode(n);
  }

  static void resetNode(const GNode& n) {
    SNode& data = graph.getData(n, Galois::MethodFlag::NONE);
    data.dist = DIST_INFINITY;
    data.done = false;
  }

  static Counts go(GNode source, bool reset) {
    using namespace Galois::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;
    
    graph.getData(source).dist = 0;
    Galois::for_each<dChunk>(source, Process(), "BFS");

    return Galois::Runtime::do_all_impl(Galois::Runtime::makeLocalRange(graph),
        CountLevels(reset), default_reduce(), true).counts;
  }

  static void init() {
    Galois::do_all_local(graph, initNode);
  }

  static void reset() {
    Galois::do_all_local(graph, resetNode);
  }
};

/**
 * The eccentricity of vertex v, ecc(v), is the greatest distance from v to any vertex.
 * A peripheral vertex v is one whose distance from some other vertex u is the
 * diameter of the graph: \exists u : dist(v, u) = D. A pseudo-peripheral vertex is a 
 * vertex v that satisfies: \forall u : dist(v, u) = ecc(v) ==> ecc(v) = ecc(u).
 *
 * Simple pseudo-peripheral algorithm:
 *  1. Choose v
 *  2. Among the vertices as far away as possible from v, select u with minimal degree
 *  3. If ecc(u) > ecc(v) then
 *       v = u and go to step 2
 *     otherwise
 *       u is the pseudo-peripheral vertex
 */
struct SimplePseudoPeripheral {
  struct min_degree {
    template<typename T>
    T operator()(const T& a, const T& b) const {
      if (!a) return b;
      if (!b) return a;
      if (graph.getData(*a).degree < graph.getData(*b).degree)
        return a;
      else
        return b;
    }
  };

  struct has_dist {
    DistType dist;
    explicit has_dist(DistType d): dist(d) { }
    boost::optional<GNode> operator()(const GNode& a) const {
      if (graph.getData(a).dist == dist)
        return boost::optional<GNode>(a);
      return boost::optional<GNode>();
    }
  };

  static std::pair<size_t, GNode> search(const GNode& start) {
     size_t ecc = BFS::go(start, false).size() - 1;
     GNode candidate =
       *Galois::ParallelSTL::map_reduce(graph.begin(), graph.end(),
           has_dist(ecc), boost::optional<GNode>(), min_degree());
     BFS::reset();
     return std::make_pair(ecc, candidate);
  }

  static GNode go(GNode source) {
    int searches = 0;

    ++searches;
    std::pair<size_t, GNode> v = search(source);
    while (true) {
      ++searches;
      std::pair<size_t, GNode> u = search(v.second);

      std::cout << "ecc(v) = " << v.first << " ecc(u) = " << u.first << "\n";
      
      if (u.first <= v.first)
        break;
      source = v.second;
      v = u;
    }

    std::cout << "Selected source: " << graph.getData(v.second)
      << " ("
      << "searches: " << searches 
      << ")\n";
    return v.second;
  }
};

/**
 * A more complicated pseudo-peripheral algorithm.
 *
 * Let the width of vertex v be the maximum number of nodes with the same
 * distance from v.
 *
 * Unlike the simple one, instead of picking a minimal degree candidate u,
 * select among some number of candidates U. Here, we select the top n
 * lowest degree nodes who do not share neighborhoods.
 *
 * If there exists a vertex u such that ecc(u) > ecc(v) proceed as in the
 * simple algorithm. 
 *
 * Otherwise, select the u that has least maximum width.
 */
struct PseudoPeripheral {
  struct order_by_degree {
    bool operator()(const GNode& a, const GNode& b) const {
      return graph.getData(a).degree < graph.getData(b).degree;
    }
  };

  //! Collect nodes with dist == d
  struct collect_nodes {
    Galois::InsertBag<GNode>& bag;
    size_t dist;
    
    collect_nodes(Galois::InsertBag<GNode>& b, size_t d): bag(b), dist(d) { }

    void operator()(const GNode& n) {
      if (graph.getData(n).dist == dist)
        bag.push(n);
    }
  };

  struct select_candidates {
    static std::deque<GNode> go(unsigned topn, size_t dist) {
      Galois::InsertBag<GNode> bag;
      Galois::do_all_local(graph, collect_nodes(bag, dist));

      // Incrementally sort nodes until we find least N who are not neighbors
      // of each other
      std::deque<GNode> nodes;
      std::deque<GNode> result;
      std::copy(bag.begin(), bag.end(), std::back_inserter(nodes));
      size_t cur = 0;
      size_t size = nodes.size();
      size_t delta = topn * 5;

      for (std::deque<GNode>::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ) {
        std::deque<GNode>::iterator mi = ii;
        if (cur + delta < size) {
          std::advance(mi, delta);
          cur += delta;
        } else {
          mi = ei;
          cur = size;
        }

        std::partial_sort(ii, mi, ei, order_by_degree());

        for (std::deque<GNode>::iterator jj = ii; jj != mi; ++jj) {
          GNode n = *jj;

          // Ignore marked neighbors
          if (graph.getData(n).done)
            continue;

          result.push_back(n);
          
          if (result.size() == topn) {
            return result;
          }

          // Mark neighbors
          for (Graph::edge_iterator nn = graph.edge_begin(n), en = graph.edge_end(n); nn != en; ++nn)
            graph.getData(graph.getEdgeDst(nn)).done = true;
        }

        ii = mi;
      }

      return result;
    }
  };

  struct Result {
    size_t ecc;
    std::deque<GNode> candidates;
    size_t max_width;
  };

  static Result search(const GNode& start) {
    Result ret;

    BFS::Counts counts = BFS::go(start, false);
    ret.max_width = *std::max_element(counts.begin(), counts.end());
    ret.ecc = counts.size() - 1;
    ret.candidates = select_candidates::go(5, ret.ecc);  
    BFS::reset();

    return ret;
  }

  static GNode go(GNode source) {
    int skips = 0;
    int searches = 0;

    ++searches;
    Result v = search(source);
    while (true) {
      std::cout 
        << "(ecc(v), max_width) =" 
        << " (" << v.ecc << ", " << v.max_width << ")"
        << " (ecc(u), max_width(u)) =";

      size_t last = std::numeric_limits<size_t>::max();
      auto ii = v.candidates.begin();
      auto ei = v.candidates.end();

      for (; ii != ei; ++ii) {
        ++searches;
        Result u = search(*ii);

        std::cout << " (" << u.ecc << ", " << u.max_width << ")";

        if (u.max_width >= last) {
          ++skips;
          continue;
        }

        if (u.ecc > v.ecc) {
          source = *ii;
          v = u;
          break;
        } else if (u.max_width < last) {
          last = u.max_width;
          source = *ii;
        }
      }

      std::cout << "\n";

      if (ii == ei)
        break;
    }

    std::cout << "Selected source: " << graph.getData(source)
      << " (skips: " << skips
      << ", searches: " << searches 
      << ")\n";
    return source;
  }
};


struct CuthillUnordered {
  template<typename RO, typename WO>
  struct PlaceFn {
    BFS::Counts& counts;
    RO& read_offset;
    WO& write_offset;

    PlaceFn(BFS::Counts& c, RO& r, WO& w): counts(c), read_offset(r), write_offset(w) { }

    void operator()(unsigned me, unsigned int tot) const {
      DistType n = me;
      while (n < counts.size()) {
	unsigned start = read_offset[n];
	unsigned t_wo = write_offset[n+1].data;
	volatile unsigned* endp = (volatile unsigned*)&write_offset[n].data;
	unsigned cend;
	unsigned todo = counts[n];
	while (todo) {
	  //spin
	  while (start == (cend = *endp)) { Galois::Runtime::LL::asmPause(); }
	  while (start != cend) {
	    GNode next = perm[start];
	    unsigned t_worig = t_wo;
	    //find eligible nodes
	    //prefetch?
	    if (0) {
	      if (start + 1 < cend) {
		GNode nnext = perm[start+1];
		for (Graph::edge_iterator ii = graph.edge_begin(nnext, Galois::MethodFlag::NONE),
		       ei = graph.edge_end(nnext, Galois::MethodFlag::NONE); ii != ei; ++ii) {
		  GNode dst = graph.getEdgeDst(ii);
		  SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
		  __builtin_prefetch(&ddata.done);
		  __builtin_prefetch(&ddata.dist);
		}
	      }
	    }
	    for (Graph::edge_iterator ii = graph.edge_begin(next, Galois::MethodFlag::NONE),
		   ei = graph.edge_end(next, Galois::MethodFlag::NONE); ii != ei; ++ii) {
	      GNode dst = graph.getEdgeDst(ii);
	      SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);
	      if (!ddata.done && (ddata.dist == n + 1)) {
		ddata.done = true;
		perm[t_wo] = dst;
		++t_wo;
	      }
	    }
	    //sort to get cuthill ordering
	    std::sort(&perm[t_worig], &perm[t_wo], sortDegFn());
	    //output nodes
	    Galois::Runtime::LL::compilerBarrier();
	    write_offset[n+1].data = t_wo;
	    //	++read_offset[n];
	    //	--level_count[n];
	    ++start;
	    --todo;
	  }
	}
	n += tot;
      }
    }
  };

  template<typename RO, typename WO>
  static void place_nodes(BFS::Counts& c, RO& read_offset, WO& write_offset) {
    Galois::on_each(PlaceFn<RO,WO>(c, read_offset, write_offset), "place");
  }

  static void place_nodes(GNode source, BFS::Counts& counts) {
    std::deque<unsigned int> read_offset;
    std::deque<Galois::Runtime::LL::CacheLineStorage<unsigned int> > write_offset;

    read_offset.push_back(0);
    std::partial_sum(counts.begin(), counts.end(), back_inserter(read_offset));
    write_offset.insert(write_offset.end(), read_offset.begin(), read_offset.end());

    perm[0] = source;
    write_offset[0].data = 1;

    place_nodes(counts, read_offset, write_offset);
  }

  static void go(GNode source) {
    BFS::Counts counts = BFS::go(source, false);
    place_nodes(source, counts);
  }
};

} // end anonymous

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer itimer("InitTime");
  itimer.start();
  graph.structureFromFile(filename);
  {
    size_t id = 0;
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii)
      graph.getData(*ii).id = id++;
  }
  BFS::init();
  itimer.stop();

  std::cout << "read " << std::distance(graph.begin(), graph.end()) << " nodes\n";
  perm.resize(std::distance(graph.begin(), graph.end()));

  Galois::StatTimer T;
  T.start();
  Galois::StatTimer Tpseudo("PseudoTime");
  Tpseudo.start();
  GNode source = useSimplePseudo ? 
      SimplePseudoPeripheral::go(*graph.begin()) : PseudoPeripheral::go(*graph.begin());
  Tpseudo.stop();

  Galois::StatTimer Tcuthill("CuthillTime");
  Tcuthill.start();
  CuthillUnordered::go(source);
  Tcuthill.stop();
  T.stop();

  std::cout << "done!\n";
  return 0;
}
