#include "galois/graphs/ASYNC_CSR_Graph.h"
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

    for (auto ii = *graph.begin(), ei = *graph.end(); 
            ii != ei; ii++) {
        std::cout << "Current Node: " << ii << std::endl;
        for (auto ne : graph.edges(ii)) {
            GNode neigh = graph.getEdgeDst(ne);
            auto ddata = graph.getEdgeData(ne);
            std::cout << " ," << neigh << ":" << ddata <<",";
        }
        std::cout << std::endl;
    }


    return 0;
}
