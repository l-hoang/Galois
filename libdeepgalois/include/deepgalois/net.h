/**
 * Based on the net.hpp file from Caffe deep learning framework.
 */
#ifndef _MODEL_H_
#define _MODEL_H_

#include <random>
#include "galois/Timer.h"
#include "deepgalois/types.h"
#include "deepgalois/gtypes.h"
#include "deepgalois/layers/graph_conv_layer.h"
#include "deepgalois/layers/softmax_loss_layer.h"
#include "deepgalois/layers/sigmoid_loss_layer.h"
#include "deepgalois/optimizer.h"
#ifndef GALOIS_USE_DIST
#include "deepgalois/context.h"
#else
#include "deepgalois/DistContext.h"
#endif

namespace deepgalois {

// N: number of vertices, D: feature vector dimentions,
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
class Net {
public:
  Net() : is_single_class(true), num_samples(0), num_classes(0),
          num_conv_layers(0), num_layers(0), num_epochs(0),
          learning_rate(0.0), dropout_rate(0.0), weight_decay(0.0),
          train_begin(0), train_end(0), train_count(0),
          val_begin(0), val_end(0), val_count(0),
          test_begin(0), test_end(0), test_count(0),
          train_masks(NULL), val_masks(NULL), test_masks(NULL), context(NULL) {}
  void init(std::string dataset_str, unsigned num_conv, unsigned epochs,
            unsigned hidden1, float lr, float dropout, float wd,
            bool selfloop, bool is_single, Graph* dGraph);
  size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
  size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id + 1]; }
  size_t get_nnodes() { return num_samples; }
  void construct_layers();
  void append_out_layer(size_t layer_id);
  void train(optimizer* opt, bool need_validate); // training
  double evaluate(std::string type, acc_t& loss, acc_t& acc); // inference
  void read_test_masks(std::string dataset, Graph* dGraph);
  acc_t fprop(size_t begin, size_t end, size_t count, mask_t* masks); // forward propagation

  //! Add a convolution layer to the network
  void append_conv_layer(size_t layer_id, bool act = false, bool norm = true,
                         bool bias = false, bool dropout = true);

  //! Save the context object to all layers of the network
  void set_contexts() {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->set_context(context);
  }
  //! set netphases for all layers in this network
  void set_netphases(deepgalois::net_phase phase) {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->set_netphase(phase);
  }
  //! print all layers
  void print_layers_info() {
    for (size_t i = 0; i < num_layers; i++)
      layers[i]->print_layer_info();
  }

  // back propogation
  void bprop() {
    for (size_t i = num_layers; i != 0; i--) {
      layers[i - 1]->backward();
    }
  }

  // update trainable weights after back-propagation
  void update_weights(optimizer* opt) {
    for (size_t i = 0; i < num_layers; i++) {
      if (layers[i]->trainable()) {
        layers[i]->update_weight(opt);
      }
    }
  }

protected:
  bool is_single_class;              // single-class (one-hot) or multi-class label
  size_t num_samples;                // number of samples: N
  size_t num_classes;                // number of vertex classes: E
  size_t num_conv_layers;            // number of convolutional layers
  size_t num_layers;                 // total number of layers (conv + output)
  unsigned num_epochs;               // number of epochs
  float learning_rate;               // learning rate
  float dropout_rate;                // dropout rate
  float weight_decay;                // weighti decay for over-fitting
  size_t train_begin, train_end, train_count;
  size_t val_begin, val_end, val_count;
  size_t test_begin, test_end, test_count;

  mask_t* train_masks;               // masks for training
  mask_t* d_train_masks;             // masks for training on device
  mask_t* val_masks;                 // masks for validation
  mask_t* d_val_masks;               // masks for validation on device
  mask_t* test_masks;                // masks for test
  mask_t* d_test_masks;              // masks for test on device
  std::vector<size_t> feature_dims;  // feature dimnesions for each layer
  std::vector<layer*> layers;        // all the layers in the neural network
#ifndef GALOIS_USE_DIST
  deepgalois::Context* context;
#else
  deepgalois::DistContext* context;
#endif

  // comparing outputs with the ground truth (labels)
#ifdef CPU_ONLY
  acc_t masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, Graph* dGraph);
  acc_t masked_multi_class_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, Graph* dGraph);
#else
  acc_t masked_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, CSRGraph *gGraph);
  acc_t masked_multi_class_accuracy(size_t begin, size_t end, size_t count, mask_t* masks, CSRGraph *gGraph);
#endif
};

} // namespace deepgalois

#endif