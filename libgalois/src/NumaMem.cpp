/** Large Allocatoins -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Loc Hoang <l_hoang@utexas.edu> (largeMallocSpecified + helpers)
 */

#include "Galois/Substrate/NumaMem.h"
#include "Galois/Substrate/PageAlloc.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/gIO.h"

#include <cassert>

using namespace Galois::Substrate;

/* Access pages on each thread so each thread has some pages already loaded 
 * (preferably ones it will use) */
static void pageIn(void* _ptr, size_t len, size_t pageSize, 
                   unsigned numThreads, bool finegrained) {
  char* ptr = static_cast<char*>(_ptr);

  if (numThreads == 1) {
    for (size_t x = 0; x < len; x += pageSize / 2)
      ptr[x] = 0;
  } else {
    getThreadPool().run(numThreads, 
     [ptr, len, pageSize, numThreads, finegrained] () 
      {
        auto myID = ThreadPool::getTID();

        if (finegrained) {
          // round robin page distribution among threads (e.g. thread 0 gets
          // a page, then thread 1, then thread n, then back to thread 0 and 
          // so on until the end of the region)
          for (size_t x  = pageSize * myID; x < len; x += pageSize * numThreads)
            ptr[x] = 0;
        } else {
          // sectioned page distribution (e.g. thread 0 gets first chunk, thread
          // 1 gets next chunk, ... last thread gets last chunk)
          for (size_t x = myID * len / numThreads; 
               x < len && x < (myID + 1) * len / numThreads; 
               x += pageSize)
            ptr[x] = 0;
        }
      }
    );
  }
}

/**
 * Causes each thread to page in a specified region of the provided memory
 * based on some distribution of elements as specified by a provided array.
 *
 * @tparam RangeArrayTy Type of threadRanges array: should either be uint32_t*
 * or uint64_t*
 * @param _ptr Pointer to the memory to page in
 * @param len Length of the memory passed in
 * @param pageSize Size of a page
 * @param numThreads Number of threads to split work amongst
 * @param threadRanges Array that specifies distribution of elements among 
 * threads
 * @param elementSize Size of an element that is to be distributed among 
 * threads
 */
template<typename RangeArrayTy>
static void pageInSpecified(void* _ptr, size_t len, size_t pageSize, 
                   unsigned numThreads, RangeArrayTy threadRanges,
                   size_t elementSize) {
  assert(numThreads > 0);
  assert(elementSize > 0);

  char* ptr = static_cast<char*>(_ptr);

  if (numThreads > 1) {
    getThreadPool().run(numThreads, 
      [ptr, len, pageSize, numThreads, threadRanges, elementSize] () {
        auto myID = ThreadPool::getTID();

        uint64_t beginLocation = threadRanges[myID];
        uint64_t endLocation = threadRanges[myID + 1];

        assert(beginLocation <= endLocation);

        //printf("[%u] begin location %u and end location %u\n", myID,
        //       beginLocation, endLocation);

        // if equal, then no memory needed to allocate in first place
        if (beginLocation != endLocation) {
          size_t beginByte = beginLocation * elementSize;
          size_t endByte;

          if (endLocation != 0) {
            // -1 since end * element will result in the first byte of the
            // next element
            endByte = (endLocation * elementSize) - 1;
          } else {
            endByte = 0;
          }

          assert(beginByte <= endByte);

          //memset(ptr + beginByte, 0, (endByte - beginByte + 1));

          uint32_t beginPage = beginByte / pageSize;
          uint32_t endPage = endByte / pageSize;

          assert(beginPage <= endPage);

          //printf("thread %u gets begin page %u and end page %u\n", myID,
          //        beginPage, endPage);

          // write a byte to every page this thread occupies
          for (uint32_t i = beginPage; i <= endPage; i++) {
            ptr[i * pageSize] = 0;
          }
        }
      }
    );
  } else {
    // 1 thread case
    for (size_t x = 0; x < len; x += pageSize / 2)
      ptr[x] = 0;
  }
}

static void largeFree(void* ptr, size_t bytes) {
  freePages(ptr, bytes/allocSize());
}

void Galois::Substrate::detail::largeFreer::operator()(void* ptr) const {
  largeFree(ptr, bytes);
}

// round data to a multiple of mult
static size_t roundup (size_t data, size_t mult) {
  auto rem = data % mult;

  if (!rem)
    return data;
  return data + (mult - rem);
}

LAptr Galois::Substrate::largeMallocInterleaved(size_t bytes, unsigned numThreads) {
  // round up to hugePageSize
  bytes = roundup(bytes, allocSize());

#ifdef GALOIS_USE_NUMA
  // We don't use numa_alloc_interleaved_subset because we really want huge 
  // pages
  // yes this is a comment in a ifdef, but if libnuma improves, this is where 
  // the alloc would go
#endif
  // Get a non-prefaulted allocation
  void* data = allocPages(bytes/allocSize(), false);

  // Then page in based on thread number
  if (data)
    // true = round robin paging
    pageIn(data, bytes, allocSize(), numThreads, true);

  return LAptr{data, detail::largeFreer{bytes}};
}

LAptr Galois::Substrate::largeMallocLocal(size_t bytes) {
  // round up to hugePageSize
  bytes = roundup(bytes, allocSize());
  // Get a prefaulted allocation
  return LAptr{allocPages(bytes/allocSize(), true), detail::largeFreer{bytes}};
}

LAptr Galois::Substrate::largeMallocFloating(size_t bytes) {
  // round up to hugePageSize
  bytes = roundup(bytes, allocSize());
  // Get a non-prefaulted allocation
  return LAptr{allocPages(bytes/allocSize(), false), detail::largeFreer{bytes}};
}

LAptr Galois::Substrate::largeMallocBlocked(size_t bytes, unsigned numThreads) {
  // round up to hugePageSize
  bytes = roundup(bytes, allocSize());
  // Get a non-prefaulted allocation
  void* data = allocPages(bytes/allocSize(), false);
  if (data)
    // false = blocked paging
    pageIn(data, bytes, allocSize(), numThreads, false);
  return LAptr{data, detail::largeFreer{bytes}};
}

/** 
 * Allocates pages for some specified number of bytes, then does NUMA page
 * faulting based on a specified distribution of elements among threads.
 *
 * @tparam RangeArrayTy Type of threadRanges array: should either be uint32_t*
 * or uint64_t*
 * @param bytes Number of bytes to allocate
 * @param numThreads Number of threads to page in regions for
 * @param threadRanges Array specifying distribution of elements among threads
 * @param elementSize Size of a data element that will be stored in the 
 * allocated memory
 * @returns The allocated memory along with a freer object
 */
template<typename RangeArrayTy>
LAptr Galois::Substrate::largeMallocSpecified(size_t bytes, 
          uint32_t numThreads, RangeArrayTy& threadRanges, 
          size_t elementSize) {
  // ceiling to nearest page
  bytes = roundup(bytes, allocSize());

  void* data = allocPages(bytes / allocSize(), false);

  // NUMA aware page in based on element distribution specified in threadRanges
  if (data) 
    pageInSpecified(data, bytes, allocSize(), numThreads, threadRanges, 
                    elementSize);

  return LAptr{data, detail::largeFreer{bytes}};
}
// Explicit template declarations since the template is defined in the .h
// file
template
LAptr Galois::Substrate::largeMallocSpecified<std::vector<uint32_t> >(size_t bytes, 
          uint32_t numThreads, std::vector<uint32_t>& threadRanges, 
          size_t elementSize);
template
LAptr Galois::Substrate::largeMallocSpecified<std::vector<uint64_t> >(size_t bytes, 
          uint32_t numThreads, std::vector<uint64_t>& threadRanges, 
          size_t elementSize);