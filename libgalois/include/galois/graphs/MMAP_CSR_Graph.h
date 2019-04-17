/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_GRAPH__MMAP_CSR_GRAPH_H
#define GALOIS_GRAPH__MMAP_CSR_GRAPH_H

#include "galois/Galois.h"
#include "galois/graphs/OfflineGraph.h"
#include "galois/graphs/Details.h"
#include "galois/graphs/GraphHelpers.h"

#include <type_traits>

namespace galois {
namespace graphs {
/**
 * MMAP'd CSR graph.
 *
 * @tparam NodeTy data on nodes
 * @tparam EdgeTy data on out edges
 */
//! [doxygennuma]
template <typename NodeTy, typename EdgeTy, bool HasNoLockable = false,
          bool UseNumaAlloc =
              false, // true => numa-blocked, false => numa-interleaved
          bool HasOutOfLineLockable = false, typename FileEdgeTy = EdgeTy>
class MMAP_CSR_Graph :
  //! [doxygennuma]
  private boost::noncopyable,
  private internal::LocalIteratorFeature<UseNumaAlloc>,
  private internal::OutOfLineLockableFeature<HasOutOfLineLockable &&
                                             !HasNoLockable> {

  // From FileGraph; tracked mmap'd regions to unmap later
  struct mapping {
    void* ptr;
    size_t len;
  };
  std::deque<mapping> mappings;

public:
  template <bool _has_id>
  struct with_id {
    typedef MMAP_CSR_Graph type;
  };

  template <typename _node_data>
  struct with_node_data {
    typedef MMAP_CSR_Graph<_node_data, EdgeTy, HasNoLockable, UseNumaAlloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _edge_data>
  struct with_edge_data {
    typedef MMAP_CSR_Graph<NodeTy, _edge_data, HasNoLockable, UseNumaAlloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _file_edge_data>
  struct with_file_edge_data {
    typedef MMAP_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                         HasOutOfLineLockable, _file_edge_data>
        type;
  };

  //! If true, do not use abstract locks in graph
  template <bool _has_no_lockable>
  struct with_no_lockable {
    typedef MMAP_CSR_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _has_no_lockable>
  using _with_no_lockable =
      MMAP_CSR_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                   HasOutOfLineLockable, FileEdgeTy>;

  //! If true, use NUMA-aware graph allocation
  template <bool _use_numa_alloc>
  struct with_numa_alloc {
    typedef MMAP_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _use_numa_alloc>
  using _with_numa_alloc =
      MMAP_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                   HasOutOfLineLockable, FileEdgeTy>;

  //! If true, store abstract locks separate from nodes
  template <bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable {
    typedef MMAP_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                         _has_out_of_line_lockable, FileEdgeTy>
        type;
  };

protected:
  typedef internal::NodeInfoBaseTypes<NodeTy,
                                      !HasNoLockable && !HasOutOfLineLockable>
      NodeInfoTypes;
  typedef internal::NodeInfoBase<NodeTy,
                                 !HasNoLockable && !HasOutOfLineLockable>
      NodeInfo;
  typedef LargeArray<NodeInfo> NodeData;

  // destinations and index data comes directly from mmap'd disk
  using EdgeIndData = LargeArray<uint64_t>;
  using EdgeDst = LargeArray<uint32_t>;
  // only works with uint32_t type data in graphs
  using EdgeData = LargeArray<uint32_t>;

public:
  typedef uint32_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef FileEdgeTy file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename NodeInfoTypes::reference node_data_reference;
  using edge_iterator =
      boost::counting_iterator<uint64_t>;
  using iterator = boost::counting_iterator<uint32_t>;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

protected:
  NodeData nodeData;
  EdgeIndData edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;

  uint64_t numNodes;
  uint64_t numEdges;
  uint32_t edgeSize;

  edge_iterator raw_begin(GraphNode N) const {
    return edge_iterator((N == 0) ? 0 : edgeIndData[N - 1]);
  }

  edge_iterator raw_end(GraphNode N) const {
    return edge_iterator(edgeIndData[N]);
  }

  template <bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    galois::runtime::acquire(&nodeData[N], mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<_A2>::type* = 0) {}

  size_t getId(GraphNode N) { return N; }

  GraphNode getNode(size_t n) { return n; }

public:
  MMAP_CSR_Graph(MMAP_CSR_Graph&& rhs) = default;
  MMAP_CSR_Graph()                     = default;
  MMAP_CSR_Graph& operator=(MMAP_CSR_Graph&&) = default;

  /**
   * Accesses the "prefix sum" of this graph; takes advantage of the fact
   * that edge_end(n) is basically prefix_sum[n] (if a prefix sum existed +
   * if prefix_sum[0] = number of edges in node 0).
   *
   * ONLY USE IF GRAPH HAS BEEN LOADED
   *
   * @param n Index into edge prefix sum
   * @returns The value that would be located at index n in an edge prefix sum
   * array
   */
  uint64_t operator[](uint64_t n) { return *(edge_end(n)); }

  MMAP_CSR_Graph(const std::string& filename) {
    // use offline graph for metadata things
    galois::graphs::OfflineGraph g(filename);
    numNodes = g.size();
    numEdges = g.sizeEdges();
    edgeSize = g.edgeSize();

    // mmap the edge destinations and the edge indices
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) GALOIS_SYS_DIE("failed opening ", "'", filename, "', MMAP CSR");

    // offsets for mapping
    uint64_t nodeIndexOffset = 4 * sizeof(uint64_t);
    uint64_t edgeDestOffset = (4 + numNodes) * sizeof(uint64_t);
    uint64_t edgeDataOffset = (4 + numNodes) * sizeof(uint64_t) +
                              (numEdges * sizeof(uint32_t));
    // padding alignment
    edgeDataOffset = (edgeDataOffset + 7) & ~7;

    // node index offsets
    uint64_t* edgeIndDataPointer = (uint64_t*)mmap(nullptr,
                                     numNodes * sizeof(uint64_t),
                                     PROT_READ, MAP_PRIVATE, fd,
                                     nodeIndexOffset);
    if (edgeIndDataPointer == nullptr) {
      GALOIS_SYS_DIE("failed to mmap index data");
    }
    mappings.push_back({edgeIndDataPointer, numNodes * sizeof(uint64_t)});
    // save large array using copy (swaps over sizes and pointer)
    edgeIndData = LargeArray<uint64_t>(edgeIndDataPointer,
                                       numNodes * sizeof(uint64_t));

    // edge destinations
    uint32_t* edgeDstPointer = (uint32_t*)mmap(nullptr, numEdges * sizeof(uint32_t),
                                 PROT_READ, MAP_PRIVATE, fd, edgeDestOffset);
    if (edgeDstPointer == nullptr) {
      GALOIS_SYS_DIE("failed to mmap edge dst");
    }
    mappings.push_back({edgeDstPointer, numEdges * sizeof(uint32_t)});
    edgeDst = LargeArray<uint32_t>(edgeDstPointer, numEdges * sizeof(uint32_t));

    // edge data
    if (!std::is_void<EdgeTy>::value) {
      uint32_t* edgeDataPointer = (uint32_t*)mmap(nullptr, numEdges * sizeof(uint32_t),
                                 PROT_READ, MAP_PRIVATE, fd, edgeDataOffset);
      if (edgeDataPointer == nullptr) {
        GALOIS_SYS_DIE("failed to mmap edge data");
      }
      mappings.push_back({edgeDataPointer, numEdges * sizeof(uint32_t)});
      edgeData = LargeArray<uint32_t>(edgeDataPointer, numEdges * sizeof(uint32_t));
    }

    // file done, close it
    close(fd);

    // TODO/NOTE this is mmap paging code from FileGraph; unmap and use if we want
    // to experiment with NUMA
    //if (numaMap) {
    //  unsigned int numThreads   = galois::runtime::activeThreads;
    //  const size_t hugePageSize = 2 * 1024 * 1024; // 2MB

    //  void* ptr;

    //  // doesn't really matter if only 1 thread; i.e. do nothing i
    //  // that case
    //  if (numThreads != 1) {
    //    // node pointer to edge dest array
    //    ptr    = (void*)outIdx;
    //    length = numNodes * sizeof(uint64_t);

    //    pageInterleaved(ptr, length, hugePageSize, numThreads);

    //    // edge dest array
    //    ptr = (void*)outs;
    //    if (graphVersion == 1) {
    //      length = numEdges * sizeof(uint32_t);
    //    } else {
    //      // v2
    //      length = numEdges * sizeof(uint64_t);
    //    }

    //    pageInterleaved(ptr, length, hugePageSize, numThreads);

    //    // edge data (if it exists)
    //    if (sizeofEdge) {
    //      ptr    = (void*)edgeData;
    //      length = numEdges * sizeofEdge;

    //      pageInterleaved(ptr, length, hugePageSize, numThreads);
    //    }
    //  }
    //}

    // allocate memory for node and edge data
    if (UseNumaAlloc) {
      nodeData.allocateBlocked(numNodes);
      this->outOfLineAllocateBlocked(numNodes, false);
    } else {
      nodeData.allocateInterleaved(numNodes);
      this->outOfLineAllocateInterleaved(numNodes);
    }
    // construct node data
    for (size_t n = 0; n < numNodes; ++n) {
      nodeData.constructAt(n);
    }
  }

  ~MMAP_CSR_Graph() {
    // unmap edge dests and edge indices
    for (auto& m : mappings) munmap(m.ptr, m.len);

  }

  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  // note unlike LC CSR this edge data is immutable
  EdgeTy getEdgeData(edge_iterator ni,
                     MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) { return edgeDst[*ni]; }

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  const_local_iterator local_begin() const {
    return const_local_iterator(this->localBegin(numNodes));
  }

  const_local_iterator local_end() const {
    return const_local_iterator(this->localEnd(numNodes));
  }

  local_iterator local_begin() {
    return local_iterator(this->localBegin(numNodes));
  }

  local_iterator local_end() {
    return local_iterator(this->localEnd(numNodes));
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = raw_begin(N), ee = raw_end(N); ii != ee; ++ii) {
        acquireNode(edgeDst[*ii], mflag);
      }
    }
    return raw_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return raw_end(N);
  }

  runtime::iterable<NoDerefIterator<edge_iterator>>
  edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::make_no_deref_range(edge_begin(N, mflag),
                                         edge_end(N, mflag));
  }

  runtime::iterable<NoDerefIterator<edge_iterator>>
  out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return edges(N, mflag);
  }

  void deallocate() {
    // node edge data
    nodeData.destroy();
    nodeData.deallocate();
    // mappings will be destroyed by destructor
  }

  /**
   * Returns the reference to the edgeIndData LargeArray
   * (a prefix sum of edges)
   *
   * @returns reference to LargeArray edgeIndData
   */
  const EdgeIndData getEdgePrefixSum() const { return edgeIndData; }
};
} // namespace graphs
} // namespace galois

#endif
