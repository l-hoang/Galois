#ifndef _GNN_TYPES_H_
#define _GNN_TYPES_H_
#include <vector>
#include <stdint.h>

// TODO namespace

#ifdef CNN_USE_DOUBLE
typedef double float_t;
typedef double feature_t;
#else
typedef float float_t;
typedef float feature_t; // feature type
#endif
typedef std::vector<float_t> vec_t; // feature vector (1D)
typedef std::vector<vec_t>
    tensor_t; // feature vectors (2D): num_samples x feature_dim
typedef std::vector<feature_t> FV; // feature vector
typedef std::vector<FV> FV2D;      // feature vectors: num_samples x feature_dim
typedef float acc_t;               // Accuracy type
typedef uint8_t label_t;  // label is for classification (supervised learning)
typedef uint8_t mask_t; // mask is used to indicate different uses of labels:
                        // train, val, test
typedef uint32_t VertexID;

#define CHUNK_SIZE 256
#define TB_SIZE 256
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NUM_CLASSES 64
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define USE_CUSPARSE


#ifdef GALOIS_USE_DIST
namespace deepgalois {
  //! Set this to let sync struct know where to get data from
  static float_t* _dataToSync = nullptr;
  //! Set this to let sync struct know the size of the vector to use during
  //! sync
  static long unsigned _syncVectorSize = 0;
}
#endif

#endif
