#pragma once

#include "galois/Logging.h"
#include "galois/GNNTypes.h"
#include <mkl.h>
#include <cmath>

namespace galois {

//! Find max index in a vector of some length
size_t MaxIndex(const size_t length, const GNNFloat* vector);
//! Given 2 float array pointers, do element wise addition of length elements
//! Can be called in parallel sections as its sigle threaded code
void VectorAdd(size_t length, const GNNFloat* a, const GNNFloat* b,
               GNNFloat* output);

//! Does a softmax operation on the input vector and saves result to output
//! vector; single threaded so it can be called in a parallel section
void GNNSoftmax(const size_t vector_length, const GNNFloat* input,
                GNNFloat* output);
//! Get derivative of softmax given the forward pass's input, the derivative
//! from loss calculation, and a temp vector to store intermediate results.
//! Everything is the same size.
void GNNSoftmaxDerivative(const size_t vector_length,
                          const GNNFloat* prev_output,
                          const GNNFloat* prev_output_derivative,
                          GNNFloat* temp_vector, GNNFloat* output);
//! Performs cross entropy given a ground truth and input and returns the loss
//! value.
template <typename TruthType>
galois::GNNFloat GNNCrossEntropy(const size_t vector_length,
                                 const TruthType* ground_truth,
                                 const GNNFloat* input) {
  GNNFloat loss = 0.0;

  // Note that this function works if there are multiple non-zeros in the
  // ground truth vector
  // If there is only 1 then this function is overkill and it should break
  // early (i.e. single class)
  // Multiclass = fine
  for (size_t i = 0; i < vector_length; i++) {
    if (ground_truth[i] == 0.0) {
      if (input[i] == 1.0) {
        loss -= std::log(static_cast<GNNFloat>(1e-10));
      } else {
        loss -= std::log(1 - input[i]);
      }
    } else {
      if (input[i] == 0.0) {
        loss -= std::log(static_cast<GNNFloat>(1e-10));
      } else {
        loss -= std::log(input[i]);
      }
    }
  }

  return loss;
}

//! Derivative of cross entropy; gradients saved into an output vector.
template <typename TruthType>
void GNNCrossEntropyDerivative(const size_t vector_length,
                               const TruthType* ground_truth,
                               const GNNFloat* input, GNNFloat* gradients) {
  for (size_t i = 0; i < vector_length; i++) {
    // TODO(loc) assumption: binary classifier, make explicit in function name
    if (ground_truth[i]) {
      gradients[i] = -1.0 / (input[i] + static_cast<float>(1e-10));
    } else {
      if (input[i] == 1.0) {
        // opposite
        gradients[i] = 1.0 / static_cast<float>(1e-10);
      } else {
        gradients[i] = 1.0 / (1.0 - input[i]);
      }
    }
  }
}

//! Calls into a library BLAS call to do matrix muliply; uses default alpha/beta
void CBlasSGEMM(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                size_t input_rows, size_t input_columns, size_t output_columns,
                const GNNFloat* a, const GNNFloat* b, GNNFloat* output);

} // namespace galois
