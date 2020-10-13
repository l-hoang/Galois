#include "galois/Logging.h"
#include "galois/GraphNeuralNetwork.h"
#include "galois/layers/GraphConvolutionalLayer.h"

int main() {
  galois::DistMemSys G;

  if (galois::runtime::getSystemNetworkInterface().Num == 1) {
    GALOIS_LOG_ERROR("This test should be run with multiple hosts/processes");
    exit(1);
  }

  auto test_graph = std::make_unique<galois::graphs::GNNGraph>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true);

  // print edges for sanity
  for (size_t node = 0; node < test_graph->size(); node++) {
    for (auto e = test_graph->EdgeBegin(node); e != test_graph->EdgeEnd(node);
         e++) {
      galois::gPrint(test_graph->host_prefix(), "Edge ",
                     test_graph->GetGID(node), " ",
                     test_graph->GetGID(test_graph->EdgeDestination(e)), "\n");
    }
  }

  // create same layer from convlayer-test and make sure result is the same even
  // in multi-host environment
  galois::GNNLayerDimensions dimension_0{.input_rows     = test_graph->size(),
                                         .input_columns  = 3,
                                         .output_columns = 2};

  // create the layer, no norm factor
  // note layer number is 1 so that it does something in backward phase
  std::unique_ptr<galois::GraphConvolutionalLayer> layer_0 =
      std::make_unique<galois::GraphConvolutionalLayer>(
          0, *(test_graph.get()), dimension_0,
          galois::GNNConfig{.allow_aggregate_after_update = false});
  layer_0->InitAllWeightsTo1();
  // make sure it runs in a sane manner
  const std::vector<galois::GNNFloat>& layer_0_forward_output =
      layer_0->ForwardPhase(test_graph->GetLocalFeatures());

  //////////////////////////////////////////////////////////////////////////////
  // sanity check output
  //////////////////////////////////////////////////////////////////////////////

  // check each row on each host: convert row into GID, and based on GID we
  // know what the ground truth is
  // row 0 = 3
  // row 1 = 6
  // row 2 = 12
  // row 3 = 18
  // row 4 = 24
  // row 5 = 30
  // row 6 = 15

  // row should correspond to LID
  for (size_t row = 0; row < test_graph->size(); row++) {
    // row -> GID
    size_t global_row = test_graph->GetGID(row);

    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
      ground_truth = 3;
      break;
    case 1:
      ground_truth = 6;
      break;
    case 2:
      ground_truth = 12;
      break;
    case 3:
      ground_truth = 18;
      break;
    case 4:
      ground_truth = 24;
      break;
    case 5:
      ground_truth = 30;
      break;
    case 6:
      ground_truth = 15;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }

    // size 2 columns
    for (size_t c = 0; c < 2; c++) {
      GALOIS_LOG_ASSERT(layer_0_forward_output[row * 2 + c] == ground_truth);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  std::vector<galois::GNNFloat> dummy_ones(test_graph->size() * 2, 1);
  // backward pass checking
  // layer 0 means that an empty weight matrix is returned since there is no
  // point passing back anything
  std::vector<galois::GNNFloat>* layer_0_backward_output =
      layer_0->BackwardPhase(test_graph->GetLocalFeatures(), &dummy_ones);

  //////////////////////////////////////////////////////////////////////////////
  // sanity check layer 0 backward output; all 0 because layer 0
  //////////////////////////////////////////////////////////////////////////////
  // since norm factors aren't invovled it is possible to do full assertions
  GALOIS_LOG_ASSERT(layer_0_backward_output->size() == test_graph->size() * 3);
  for (size_t i = 0; i < layer_0_backward_output->size(); i++) {
    GALOIS_LOG_ASSERT((*layer_0_backward_output)[i] == 0);
  }

  //////////////////////////////////////////////////////////////////////////////
  // layer 1 to check backward output
  //////////////////////////////////////////////////////////////////////////////
  std::unique_ptr<galois::GraphConvolutionalLayer> layer_1 =
      std::make_unique<galois::GraphConvolutionalLayer>(
          1, *(test_graph.get()), dimension_0,
          galois::GNNConfig{.allow_aggregate_after_update = false});
  layer_1->InitAllWeightsTo1();
  const std::vector<galois::GNNFloat>& layer_1_forward_output =
      layer_1->ForwardPhase(test_graph->GetLocalFeatures());

  // same check for forward as before
  for (size_t row = 0; row < test_graph->size(); row++) {
    // row -> GID
    size_t global_row = test_graph->GetGID(row);

    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
      ground_truth = 3;
      break;
    case 1:
      ground_truth = 6;
      break;
    case 2:
      ground_truth = 12;
      break;
    case 3:
      ground_truth = 18;
      break;
    case 4:
      ground_truth = 24;
      break;
    case 5:
      ground_truth = 30;
      break;
    case 6:
      ground_truth = 15;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }

    // size 2 columns
    for (size_t c = 0; c < 2; c++) {
      GALOIS_LOG_ASSERT(layer_1_forward_output[row * 2 + c] == ground_truth);
    }
  }

  // since layer isn't 0 anymore, backward phase will actually return something
  dummy_ones.assign(test_graph->size() * 2, 1);
  std::vector<galois::GNNFloat>* layer_1_backward_output =
      layer_1->BackwardPhase(test_graph->GetLocalFeatures(), &dummy_ones);

  for (size_t row = 0; row < test_graph->size(); row++) {
    // row -> GID
    size_t global_row = test_graph->GetGID(row);

    galois::GNNFloat ground_truth = 0.0;

    switch (global_row) {
    case 0:
    case 6:
      ground_truth = 2;
      break;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
      ground_truth = 4;
      break;
    default:
      GALOIS_LOG_FATAL("bad global row for test graph");
      break;
    }

    // size 3 columns
    for (size_t c = 0; c < 3; c++) {
      GALOIS_LOG_ASSERT((*layer_1_backward_output)[row * 3 + c] ==
                        ground_truth);
    }
  }
}
