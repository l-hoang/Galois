/** Thread pool common implementation -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/EnvCheck.h"
#include "Galois/Substrate/HWTopo.h"
#include "Galois/Substrate/gio.h"

#include <algorithm>
#include <iostream>

// Forward declare this to avoid including PerThreadStorage.
// We avoid this to stress that the thread Pool MUST NOT depend on PTS.
namespace Galois {
namespace Substrate {

extern void initPTS(unsigned);

}
}


using namespace Galois::Substrate;

thread_local ThreadPool::per_signal ThreadPool::my_box;


ThreadPool::ThreadPool() 
  : mi(getHWTopo().first),
    reserved(0),
    masterFastmode(false), 
    running(false)
{
  signals.resize(mi.maxThreads);
  initThread(0);

  for (unsigned i = 1; i < mi.maxThreads; ++i) {
    std::thread t(&ThreadPool::threadLoop, this, i);
    threads.emplace_back(std::move(t));
  }

  //we don't want signals to have to contain atomics, since they are set once
  while (std::any_of(signals.begin(), signals.end(), [](per_signal* p) { return !p || !p->done; })) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }
}

ThreadPool::~ThreadPool() {
  destroyCommon();
  for(auto& t : threads)
    t.join();
}

void ThreadPool::destroyCommon() {
  beKind(); // reset fastmode
  run(mi.maxThreads, []() { throw shutdown_ty(); });
}

void ThreadPool::burnPower(unsigned num) {
  //changing number of threads?  just do a reset
  if (masterFastmode && masterFastmode != num)
    beKind();
  if (!masterFastmode) {
    run(num, []() { throw fastmode_ty{true}; });
    masterFastmode = num;
  }
}

void ThreadPool::beKind() {
  if (masterFastmode) {
    run(masterFastmode, []() { throw fastmode_ty{false}; });
    masterFastmode = 0;
  }
}

//inefficient append
template<typename T>
static void atomic_append(std::atomic<T*>& headptr, T* newnode) {
  T* n = nullptr;
  if (!headptr.compare_exchange_strong(n, newnode))
    atomic_append(headptr.load()->next, newnode);
}

//find id
template<typename T> 
static unsigned findID(std::atomic<T*>& headptr, T* node, unsigned off) {
  T* n = headptr.load();
  assert(n);
  if (n == node)
    return off;
  else
    return findID(n->next, node, off+1);
}

template<typename T>
static T* getNth(std::atomic<T*>& headptr, unsigned off) {
  T* n = headptr.load();
  if (!off)
    return n;
  else
    return getNth(n->next, off - 1);
}

void ThreadPool::initThread(unsigned tid) {
  signals[tid] = &my_box;
  my_box.topo = getHWTopo().second[tid];
  // Initialize 
  Substrate::initPTS(mi.maxThreads);

  if (!EnvCheck("GALOIS_DO_NOT_BIND_THREADS"))
    if (my_box.topo.tid != 0 || !EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD"))
      bindThreadSelf(my_box.topo.osContext);
  my_box.done = true;
}

void ThreadPool::threadLoop(unsigned tid) {
  initThread(tid);
  bool fastmode = false;
  auto& me = my_box;
  do {
    me.wait(fastmode);
    cascade(fastmode);
    try {
      work();
    } catch (const shutdown_ty&) {
      return;
    } catch (const fastmode_ty& fm) {
      fastmode = fm.mode;
    } catch (const dedicated_ty dt) {
      me.done = 1;
      dt.fn();
      return;
    } catch (const std::exception &exc) {
      // catch anything thrown within try block that derives from std::exception
      std::cerr << exc.what();
      abort();
    }
    catch (...) {
      abort();
    }
    decascade();
  } while (true);
}


void ThreadPool::decascade() {
  auto& me = my_box;
  //nothing to wake up
  if (me.wbegin != me.wend) {
    auto midpoint = me.wbegin + (1 + me.wend - me.wbegin)/2;
    auto& c1done = signals[me.wbegin]->done;
    while (!c1done) { asmPause(); }
    if (midpoint < me.wend) {
      auto& c2done = signals[midpoint]->done;
      while (!c2done) { asmPause(); }
    }
  }
  me.done = 1;
}

void ThreadPool::cascade(bool fastmode) {
  auto& me = my_box;
  assert(me.wbegin <= me.wend);

  //nothing to wake up
  if (me.wbegin == me.wend)
    return;

  auto midpoint = me.wbegin + (1 + me.wend - me.wbegin)/2;
  // static std::mutex m;
  // {
  //   std::lock_guard<std::mutex> lg(m);
  //   std::cout << getTID() << "\t" << me.wbegin << ": (" << me.wbegin + 1 << " " << midpoint << ")\t" << midpoint << ": (" << midpoint+1 << " " << me.wend << ")\n";
  // }

  auto child1 = signals[me.wbegin];
  child1->wbegin = me.wbegin+1;
  child1->wend = midpoint;
  child1->wakeup(fastmode);
  
  if (midpoint < me.wend) {
    auto child2 = signals[midpoint];
    child2->wbegin = midpoint + 1;
    child2->wend = me.wend;
    child2->wakeup(fastmode);
  }
}

void ThreadPool::runInternal(unsigned num) {
  //sanitize num
  //seq write to starting should make work safe
  GALOIS_ASSERT(!running, "recursive thread pool execution not supported");
  running = true;
  num = std::min(std::max(1U,num), mi.maxThreads - reserved);
  //my_box is tid 0
  auto& me = my_box;
  me.wbegin = 1;
  me.wend = num;

  assert(!masterFastmode || masterFastmode == num);
  //launch threads
  cascade(masterFastmode);
  // Do master thread work
  try {
    work();
  } catch (const shutdown_ty&) {
    return;
  } catch (const fastmode_ty& fm) {
  }
  //wait for children
  decascade();
  // Clean up
  work = nullptr;
  running = false;
}

void ThreadPool::runDedicated(std::function<void(void)>& f) {
  GALOIS_ASSERT(!running, "can't start dedicated thread durring parallel section");
  ++reserved;
  GALOIS_ASSERT(reserved < mi.maxThreads, "Too many dedicated threads");
  work = [&f] () { throw dedicated_ty {f}; };
  auto child = signals[mi.maxThreads - reserved];
  child->wbegin = 0;
  child->wend = 0;
  child->done = 0;
  child->wakeup(masterFastmode);
  while (!child->done) { asmPause(); }
  work = nullptr;
  //FIXME: Galois::setActiveThreads(Galois::getActiveThreads());
}

ThreadPool& ThreadPool::getThreadPool() {
  static ThreadPool p;
  return p;
}