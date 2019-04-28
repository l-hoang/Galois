//#include "galois/graphs/OnDemand_CSR_Graph.h"
#include "galois/graphs/ASYNCNB_CSR_Graph.h"
#include "galois/graphs/Graph.h"
#include "galois/Galois.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>

namespace cll = llvm::cl;

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<unsigned int>
    startNode("startNode",
              cll::desc("Node to start search from (default value 0)"),
              cll::init(0));
using Graph = galois::graphs::ASYNC_CSR_Graph<uint32_t, uint32_t>::with_no_lockable<true>::type;
using GNode = Graph::GraphNode;
//using Graph = galois::graphs::LC_CSR_Graph<unsigned, unsigned>::with_no_lockable<true>::type;

int main(int argc, char** argv)
{
    galois::SharedMemSys G;
    LonestarStart(argc, argv, "practice", "practice", "practice");
    std::cout << "Before reading graph " << filename << "\n";
    Graph graph(filename);

    galois::InsertBag<uint32_t> initBag;

    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode) {
                initBag.emplace(cNode);
            }, galois::steal());

    galois::for_each(galois::iterate(initBag.begin(), initBag.end()),
            [&](GNode cNode, auto& ctx) {
                //std::cout << "target node: " << cNode << "\n";
                if (!graph.loadEdges(cNode)) {
                    //std::cout << "not loaded yet\n";
                    ctx.push(cNode);
                } else {
                    std::cout << "loading is done\n";
                    for (auto ne : graph.edges(cNode)) {
                        GNode neigh = graph.getEdgeDst(ne);
                    }
                }
            }, galois::no_conflicts(), galois::chunk_size<64u>());

   // for (int i = 0; i < 10; i++) sleep(1);

    return 0;
}
