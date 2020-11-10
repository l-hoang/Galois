#pragma once
#include "galois/GNNTypes.h"
#include "galois/graphs/GNNGraph.cuh"

namespace galois {

//! Holds pointers for GPU memory for GCN layer
class GCNGPUAllocations {
public:
  // free memory
  ~GCNGPUAllocations();
  // allocate the 3 temp arrays
  void Allocate(size_t input_elements, size_t output_elements);
  GNNFloat* in_temp_1() { return in_temp_1_; }
  GNNFloat* in_temp_2() { return in_temp_2_; }
  GNNFloat* out_temp() { return out_temp_; }

  void AggregateAllGPU(const graphs::GNNGraphGPUAllocations& gpu_graph,
                       size_t num_nodes, size_t column_length,
                       const GNNFloat* node_embeddings,
                       GNNFloat* aggregate_output);

private:
  GNNFloat* in_temp_1_{nullptr};
  GNNFloat* in_temp_2_{nullptr};
  GNNFloat* out_temp_{nullptr};
};

} // namespace galois