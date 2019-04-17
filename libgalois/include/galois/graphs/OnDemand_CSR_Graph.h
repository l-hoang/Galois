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

#ifndef GALOIS_GRAPH__ASYNC_CSR_Graph_H
#define GALOIS_GRAPH__ASYNC_CSR_Graph_H

#include "galois/Galois.h"
#include "galois/graphs/OfflineGraph.h"
#include "galois/graphs/Details.h"
#include "galois/graphs/GraphHelpers.h"
#include "galois/DynamicBitset.h"

#include <thread>
#include <csignal>
#include <type_traits>
#include <aio.h>

namespace galois {
namespace graphs {

/**
 * Asynchronous CSR graph.
 *
 * @tparam NodeTy data on nodes
 * @tparam EdgeTy data on out edges
 */
//! [doxygennuma]
template <typename NodeTy, typename EdgeTy, bool HasNoLockable = false,
          bool UseNumaAlloc =
              false, // true => numa-blocked, false => numa-interleaved
          bool HasOutOfLineLockable = false, typename FileEdgeTy = EdgeTy>
class ASYNC_CSR_Graph :
  //! [doxygennuma]
  private boost::noncopyable,
  private internal::LocalIteratorFeature<UseNumaAlloc>,
  private internal::OutOfLineLockableFeature<HasOutOfLineLockable &&
                                             !HasNoLockable> {

public:

  template <bool _has_id>
  struct with_id {
    typedef ASYNC_CSR_Graph type;
  };

  template <typename _node_data>
  struct with_node_data {
    typedef ASYNC_CSR_Graph<_node_data, EdgeTy, HasNoLockable, UseNumaAlloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _edge_data>
  struct with_edge_data {
    typedef ASYNC_CSR_Graph<NodeTy, _edge_data, HasNoLockable, UseNumaAlloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _file_edge_data>
  struct with_file_edge_data {
    typedef ASYNC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                         HasOutOfLineLockable, _file_edge_data>
        type;
  };

  //! If true, do not use abstract locks in graph
  template <bool _has_no_lockable>
  struct with_no_lockable {
    typedef ASYNC_CSR_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _has_no_lockable>
  using _with_no_lockable =
      ASYNC_CSR_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                   HasOutOfLineLockable, FileEdgeTy>;

  //! If true, use NUMA-aware graph allocation
  template <bool _use_numa_alloc>
  struct with_numa_alloc {
    typedef ASYNC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                         HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _use_numa_alloc>
  using _with_numa_alloc =
      ASYNC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                   HasOutOfLineLockable, FileEdgeTy>;

  //! If true, store abstract locks separate from nodes
  template <bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable {
    typedef ASYNC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
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

  int fd; // open file descriptor for reads later

  galois::DynamicBitSet loadStatus; // maintain unloaded/loading status.
  galois::DynamicBitSet completeStatus;// maintain complete status.

  // destinations and index data comes directly from mmap'd disk
  using EdgeIndData = LargeArray<uint64_t>;
  using EdgeDst = LargeArray<uint32_t>;
  using EdgeData = LargeArray<EdgeTy>;

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

  size_t edgeIndexSize;
  size_t edgeDestSize;
  size_t edgeDataSize;

  uint64_t nodeIndexOffset;
  uint64_t edgeDestOffset;
  uint64_t edgeDataOffset;

  std::string fName;

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
  ASYNC_CSR_Graph(ASYNC_CSR_Graph&& rhs) = default;
  ASYNC_CSR_Graph()                     = default;
  ASYNC_CSR_Graph& operator=(ASYNC_CSR_Graph&&) = default;

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


  ASYNC_CSR_Graph(const std::string& fName) : fName(fName){
    // use offline graph for metadata things
    galois::graphs::OfflineGraph g(fName);
    numNodes = g.size();
    numEdges = g.sizeEdges();
    edgeSize = g.edgeSize();

    //std::cout << "Num of nodes: " << numNodes << ", numEdges: " << numEdges << ", edgeSize: " << edgeSize << std::endl;

    // mmap the edge destinations and the edge indices
    fd = open(fName.c_str(), O_RDONLY);
    if (fd == -1) GALOIS_SYS_DIE("failed opening ", "'", fName, "', MMAP CSR");

    // each array size 
    edgeIndexSize = numNodes * sizeof(uint64_t);
    edgeDestSize = numEdges * sizeof(uint32_t);
    edgeDataSize = numEdges * sizeof(EdgeTy);
    // offsets for mapping
    nodeIndexOffset = 4 * sizeof(uint64_t);
    edgeDestOffset = (4 + numNodes) * sizeof(uint64_t);
    edgeDataOffset = (4 + numNodes) * sizeof(uint64_t) +
                              (numEdges * sizeof(uint32_t));
    // padding alignment
    edgeDataOffset = (edgeDataOffset + 7) & ~7;

    // move file descriptor to node index offsets array.
    if (nodeIndexOffset != lseek(fd, nodeIndexOffset, SEEK_SET)) {
      GALOIS_DIE("Failed to move file pointer to edge index array.");
    }

    // allocate memory for node and edge data
    if (UseNumaAlloc) {
      nodeData.allocateBlocked(numNodes);
      edgeIndData.allocateBlocked(numNodes);
      edgeDst.allocateBlocked(numEdges);
      edgeData.allocateBlocked(numEdges);
      this->outOfLineAllocateBlocked(numNodes, false);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
    // construct node data
    for (size_t n = 0; n < numNodes; ++n) {
      nodeData.constructAt(n);
    }

    // read indices for nodes (prefix sum)
    // TODO read may not read everything at once; need to put this in a while
    // loop
    if (edgeIndexSize != read(fd, edgeIndData.data(), edgeIndexSize)) {
      GALOIS_DIE("Failed to read edge index array.");
    }

    loadStatus.resize(numNodes);
    completeStatus.resize(numNodes);
  }

  ~ASYNC_CSR_Graph() {
    // file done, close it
    close(fd);
  }

  // TODO revise this
  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    if (!completeStatus.test(N) && loadStatus.test(N)) {
    // Loading request is done, but not yet complete.
    // just skip.
      // TODO add a busy wait
    } else if (!completeStatus.test(N) && !loadStatus.test(N)) {
    // Loading request does not start yet.
    // should fill neighbor index array, neighbor array, neighbor data array.
    // Each request requires respective struct aiocb.
    // (NOTE: I don't believe it would be the best option, but
    // we can improve implementations.)

      /* Phase 1: read corresponding edge index array */
      struct aiocb eDestAiocb;
      bzero((char *)(&eDestAiocb), sizeof(struct aiocb));
      eDestAiocb.aio_fildes = fd;
      eDestAiocb.aio_offset = edgeDestOffset;
      if (N == 0) {
          // Phase 1 should be initialized at the first phase,
          // since in order to update edge destination array,
          // we should know edge index range.
          eDestAiocb.aio_nbytes = edgeIndData[0]*sizeof(uint32_t);
          eDestAiocb.aio_buf = &edgeDst[0];
      //std::cout << "buf size: " << eDestAiocb.aio_nbytes << ", dest array index: " <<
      //    "0" << ", file offset:" << eDestAiocb.aio_offset <<"\n";
      //
      //    std::cout << N << " >>\n";
      //    std::cout << edgeDst[0] << "\n";
      } else {
          eDestAiocb.aio_nbytes = (edgeIndData[N]-edgeIndData[N-1])*sizeof(uint32_t);
          eDestAiocb.aio_buf = &edgeDst[edgeIndData[N-1]];
          eDestAiocb.aio_offset += (edgeIndData[N-1]*sizeof(uint32_t));

      //std::cout << "buf size: " << eDestAiocb.aio_nbytes << ", dest array index: " <<
      //    edgeIndData[N-1] << ", file offset:" << eDestAiocb.aio_offset <<"\n";

      }

      if(aio_read(&eDestAiocb) < 0) 
         printf("aio_read is failed\n");
      while (aio_error(&eDestAiocb) == EINPROGRESS) ;
      int ret;
      if ((ret = aio_return(&eDestAiocb)) > 0) {
          printf("Ret[%d] \n", ret);
          for (int i = 0; i < eDestAiocb.aio_nbytes/sizeof(uint32_t); i++)
              printf("%d, Buff[%d] \n", i, *((uint32_t *) eDestAiocb.aio_buf+i));
      }
      loadStatus.set(N);
    } else {
      NodeInfo& NI = nodeData[N];
      acquireNode(N, mflag);
      return NI.getData();
    }
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
