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
#include <type_traits>
#include <aio.h>
#include <csignal>
#include <unistd.h>
#include <sys/syscall.h>

#define SIG_AIO SIGRTMIN+1

#define AIO_VEC_SIZE 1000

//#ifndef sigev_notify_thread_id
//#define sigev_notify_thread_id _sigev_un._tid
//#endif

namespace galois {
namespace graphs {

static thread_local unsigned int aioReqIdx;

//void read_handler(sigval_t sigval)

struct aiocb_with_info {
  struct aiocb aioReq;
  int curNode;
  galois::DynamicBitSet *complStats;
  int status;
};

void readDstCallback(int s, siginfo_t * info, void *ctx) {
    struct aiocb *aioReq;
    galois::DynamicBitSet *complStats;
    int curNode;

    struct aiocb_with_info *req = (struct aiocb_with_info *)
                                       info->si_value.sival_ptr;

    aioReq = &(req->aioReq);
    complStats = req->complStats;
    curNode = req->curNode;
    //galois::gDebug("finished node is ", curNode);
    //printf("\n\nRequested aio: %p\n", aioReq);
    //printf("Requested complSets: %p\n", complStats);
    //printf("\nRequested node: %d\n", curNode);
    int aio_error_code = aio_error(aioReq);
    if (aio_error_code == 0) {
        int ret = aio_return(aioReq);
        if (ret == -1) {
          GALOIS_SYS_DIE("aio return returned bad value ", ret);
        }
        //printf("Req offset: %d, size: %d\n", aioReq->aio_offset, aioReq->aio_nbytes);
        //printf("results: %d\n", *(uint32_t *) aioReq->aio_buf);
    } else if (aio_error_code == ECANCELED) {
      galois::gPrint("error code is ECANCELED\n");
      GALOIS_DIE("handler shouldn't have got cancelled\n");
    } else if (aio_error_code == EINPROGRESS) {
      galois::gPrint("error code is EINPROGRESS\n");
      GALOIS_DIE("handler shouldn't be in progress at completion\n");
    } else {
      perror("failure in aio_error, failed to get return");
      GALOIS_DIE("aio error didn't work on completed request\n");
    }

    complStats->set_for_sure(curNode);
    //galois::gDebug(curNode, " set for sure");
}

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
  struct sigaction act;
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

  int fd; // Edge data is read by AIO.
  galois::DynamicBitSet loadStatus; // maintain unloaded/loading status.
  galois::DynamicBitSet completeStatus;// maintain complete status.
  // destinations and index data comes directly from mmap'd disk
  using EdgeIndData = LargeArray<uint64_t>;
  using EdgeDst = LargeArray<uint32_t>;
  // only works with uint32_t type data in graphs
  using EdgeData = LargeArray<EdgeTy>;
  struct aiocb* data_aiocb;
public:
  typedef uint32_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef FileEdgeTy file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename NodeInfoTypes::reference node_data_reference;
  typedef typename EdgeData::reference edge_data_reference;
  using edge_iterator =
      boost::counting_iterator<uint64_t>;
  using iterator = boost::counting_iterator<uint32_t>;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;
  typedef galois::gstl::Vector<struct aiocb_with_info> VecAioCbTy;
  typedef galois::substrate::PerThreadStorage<VecAioCbTy> ThreadLocalData;

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

  ThreadLocalData thLocDat;

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
  ASYNC_CSR_Graph(ASYNC_CSR_Graph&& rhs) = delete;
  ASYNC_CSR_Graph()                      = delete;
  ASYNC_CSR_Graph& operator=(ASYNC_CSR_Graph&&) = delete;

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


  ASYNC_CSR_Graph(const std::string& fName) {
    // use offline graph for metadata things
    galois::graphs::OfflineGraph g(fName);
    numNodes = g.size();
    numEdges = g.sizeEdges();
    edgeSize = g.edgeSize();

    // register aio signal
    act.sa_sigaction = readDstCallback;
    act.sa_flags = SA_SIGINFO;
    sigemptyset(&act.sa_mask);
    sigaction(SIG_AIO, &act, NULL);

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

    // allocate memory for node and edge data
    if (UseNumaAlloc) {
      nodeData.allocateBlocked(numNodes);
      edgeIndData.allocateBlocked(numNodes);
      edgeDst.allocateBlocked(numEdges);
      edgeData.allocateBlocked(numEdges);
      this->outOfLineAllocateBlocked(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }

    // Move file descriptor to node index offsets array.
    if (nodeIndexOffset != lseek(fd, nodeIndexOffset, SEEK_SET)) {
        GALOIS_DIE("Failed to move file pointer to edge index array.");
    }

    // Edge index array is set in initial phase.
    // Therefore, we can know which node should be read.
    if (edgeIndexSize != read(fd, edgeIndData.data(), edgeIndexSize)) {
        GALOIS_DIE("Failed to read edge index array.");
    }

    // construct node data
    for (size_t n = 0; n < numNodes; ++n) {
      nodeData.constructAt(n);
    }

    loadStatus.resize(numNodes);
    completeStatus.resize(numNodes);
    aioReqIdx = 0;
  }

  ~ASYNC_CSR_Graph() {
    // file done, close it
    close(fd);
  }

  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  bool load_edges(GraphNode N) {
      bool result = false;
      auto& aioCbVec = *thLocDat.getLocal();
      if (!aioCbVec.size()) {
        // Initialize thread local vector.
        aioCbVec.clear();
        aioCbVec.resize(AIO_VEC_SIZE);
        memset(aioCbVec.data(), 0, aioCbVec.size() * sizeof(aiocb_with_info));
      }

      if (!completeStatus.test(N)) {
          bool claim = loadStatus.set(N);
          if (claim) {
            // I claimed it, so I have to load it.
            // Read corresponding edge index array.
            // Request aio.
            struct aiocb* eDestAioCb = &(aioCbVec[aioReqIdx].aioReq);

            int ret;
            //if ((ret = aio_error(eDestAioCb)) == EINPROGRESS) {
            //   // looped around circular buffer, still in progress; design
            //   // choice, just try again later instead of going past current
            //   // position in buffer
            //   // unset load status, return false
            //   //galois::gPrint("reseting load status ", N, "\n");
            //   loadStatus.reset(N);
            //   // another thread (or same thread) will pick this back up
            //   // later...
            //   return false;
            //}

            if (aioCbVec[aioReqIdx].status == 1) {
              // wait until struct is free (i.e. complete status has been set)
              while (!completeStatus.test(aioCbVec[aioReqIdx].curNode)) {
                //galois::gDebug("loop");
              }
            }

            // store the current node.
            aioCbVec[aioReqIdx].curNode = N;
            aioCbVec[aioReqIdx].complStats = &completeStatus;
            aioCbVec[aioReqIdx].status = 1;

            memset(eDestAioCb, 0ul, sizeof(struct aiocb));
            // graph file on disk
            eDestAioCb->aio_fildes = fd;

            // setup buffers and offsets to read
            eDestAioCb->aio_offset = edgeDestOffset;
            if (N == 0) {
                eDestAioCb->aio_nbytes = edgeIndData[0]*sizeof(uint32_t);
                eDestAioCb->aio_buf = &edgeDst[0];
            } else {
                eDestAioCb->aio_nbytes = (edgeIndData[N] - edgeIndData[N - 1])
                                         * sizeof(uint32_t);
                eDestAioCb->aio_buf = &edgeDst[edgeIndData[N - 1]];
                eDestAioCb->aio_offset += (edgeIndData[N - 1] *
                                          sizeof(uint32_t));
            }

            //std::cout << "offset: " << eDestAioCb->aio_offset << ", bytes: " <<
            //       eDestAioCb->aio_nbytes << "\n";

            // get notified with signal
            eDestAioCb->aio_sigevent.sigev_notify = SIGEV_SIGNAL;
            // signal type is AIO
            eDestAioCb->aio_sigevent.sigev_signo = SIG_AIO;
            eDestAioCb->aio_sigevent.sigev_notify_attributes = NULL;
            eDestAioCb->aio_sigevent.sigev_value.sival_ptr = &(aioCbVec[aioReqIdx]);

            //galois::gDebug("before idx ", aioReqIdx);
            // increment circular buffer
            aioReqIdx++;
            if (aioReqIdx == AIO_VEC_SIZE) aioReqIdx = 0;
            //galois::gDebug("after idx ", aioReqIdx);

            //galois::gDebug(N, " made it read");
            ret = aio_read(eDestAioCb);
            if (ret != 0) {
              GALOIS_SYS_DIE("aio read failure in load_edge");
            }
          } else {
            //galois::gPrint("spin lock ", N);
          }
          // Otherwise...
          // Necessary data related to node N is already requested by another node
      } else {
        //galois::gPrint("already loaded ", N, "\n");
        // already loaded, don't need to do anything
        result= true;
      }

      return result;
  }

  // TODO currently always returns 1
  uint32_t aOne = 1;
  // note unlike LC CSR this edge data is immutable
  edge_data_reference getEdgeData(edge_iterator ni,
                     MethodFlag mflag = MethodFlag::UNPROTECTED) {
    return aOne;
    // galois::runtime::checkWrite(mflag, false);
    //return edgeData[*ni];
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
    nodeData.destroy();
    nodeData.deallocate();
    edgeIndData.destroy();
    edgeIndData.deallocate();
    edgeDst.destroy();
    edgeDst.deallocate();
    edgeData.destroy();
    edgeData.deallocate();
    //thLocDat.destroy();
    //thLocDat.deallocate();
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
