// XXX include net interface if necessary
#include "galois/Logging.h"
#include "galois/graphs/ReadGraph.h"
#include "galois/graphs/GNNGraph.h"
#include <limits>

namespace {
//! Partitions a particular dataset given some partitioning scheme
std::unique_ptr<galois::graphs::GNNGraph::GNNDistGraph>
LoadPartition(const std::string& input_directory,
              const std::string& dataset_name,
              galois::graphs::GNNPartitionScheme partition_scheme) {
  // XXX input path
  std::string input_file = input_directory + dataset_name + ".csgr";
  GALOIS_LOG_VERBOSE("Partition loading: File to read is {}", input_file);

  // load partition
  switch (partition_scheme) {
  case galois::graphs::GNNPartitionScheme::kOEC:
    return galois::cuspPartitionGraph<GnnOEC, char, void>(
        input_file, galois::CUSP_CSR, galois::CUSP_CSR, true, "", "", false, 1);
  case galois::graphs::GNNPartitionScheme::kCVC:
    return galois::cuspPartitionGraph<GnnCVC, char, void>(
        input_file, galois::CUSP_CSR, galois::CUSP_CSR, true, "", "", false, 1);
  default:
    GALOIS_LOG_FATAL("Error: partition scheme specified is invalid");
    return nullptr;
  }
}

} // end namespace

namespace galois {
namespace graphs {
GNNFloat* gnn_matrix_to_sync_            = nullptr;
size_t gnn_matrix_to_sync_column_length_ = 0;
} // namespace graphs
} // namespace galois

galois::graphs::GNNGraph::GNNGraph(const std::string& dataset_name,
                                   GNNPartitionScheme partition_scheme,
                                   bool has_single_class_label)
    : GNNGraph(galois::default_gnn_dataset_path, dataset_name, partition_scheme,
               has_single_class_label) {}

galois::graphs::GNNGraph::GNNGraph(const std::string& input_directory,
                                   const std::string& dataset_name,
                                   GNNPartitionScheme partition_scheme,
                                   bool has_single_class_label)
    : input_directory_(input_directory) {
  GALOIS_LOG_VERBOSE("[{}] Constructing partitioning for {}", host_id_,
                     dataset_name);
  // save host id
  host_id_ = galois::runtime::getSystemNetworkInterface().ID;
  host_prefix_ =
      std::string("[") +
      std::to_string(galois::runtime::getSystemNetworkInterface().ID) +
      std::string("] ");
  // load partition
  partitioned_graph_ =
      LoadPartition(input_directory_, dataset_name, partition_scheme);

  // read additional graph data
  ReadLocalLabels(dataset_name, has_single_class_label);
  ReadLocalFeatures(dataset_name);
  ReadLocalMasks(dataset_name);

  // init gluon from the partitioned graph
  sync_substrate_ =
      std::make_unique<galois::graphs::GluonSubstrate<GNNDistGraph>>(
          *partitioned_graph_, host_id_,
          galois::runtime::getSystemNetworkInterface().Num, false);

  // read in entire graph topology
  ReadWholeGraph(dataset_name);
  // init norm factors using the whole graph topology
  InitNormFactor();

#ifdef GALOIS_ENABLE_GPU
  // allocate/copy data structures over to GPU
  GALOIS_LOG_VERBOSE("[{}] Initializing GPU memory", host_id_);
  InitGPUMemory();
#endif
}

bool galois::graphs::GNNGraph::IsValidForPhase(
    const unsigned lid, const galois::GNNPhase current_phase) const {
  // convert to gid first
  size_t gid = partitioned_graph_->getGID(lid);

  // select range to use based on phase
  const GNNRange* range_to_use;
  switch (current_phase) {
  case GNNPhase::kTrain:
    range_to_use = &global_training_mask_range_;
    break;
  case GNNPhase::kValidate:
    range_to_use = &global_validation_mask_range_;
    break;
  case GNNPhase::kTest:
    range_to_use = &global_testing_mask_range_;
    break;
  default:
    GALOIS_LOG_FATAL("Invalid phase used");
    range_to_use = nullptr;
  }

  // if within range, it is valid
  // TODO there is an assumption here that ranges are contiguous; may not
  // necessarily be the case in all inputs in which case using the mask is safer
  // (but less cache efficient)
  if (range_to_use->begin <= gid && gid < range_to_use->end) {
    return true;
  } else {
    return false;
  }
}

void galois::graphs::GNNGraph::AggregateSync(
    GNNFloat* matrix_to_sync, const size_t matrix_column_size) const {
  // set globals for the sync substrate
  gnn_matrix_to_sync_               = matrix_to_sync;
  gnn_matrix_to_sync_column_length_ = matrix_column_size;

  // XXX bitset setting

  // call sync
  sync_substrate_->sync<writeSource, readAny, GNNSumAggregate>(
      "GraphAggregateSync");
}

void galois::graphs::GNNGraph::ReadLocalLabels(const std::string& dataset_name,
                                               bool has_single_class_label) {
  GALOIS_LOG_VERBOSE("[{}] Reading labels from disk...", host_id_);
  std::string filename;
  if (has_single_class_label) {
    filename = input_directory_ + dataset_name + "-labels.txt";
  } else {
    filename = input_directory_ + dataset_name + "-mlabels.txt";
  }

  // read file header, save num label classes while at it
  std::ifstream file_stream;
  file_stream.open(filename, std::ios::in);
  size_t num_nodes;
  file_stream >> num_nodes >> num_label_classes_ >> std::ws;
  assert(num_nodes == partitioned_graph_->globalSize());

  // allocate memory for labels
  if (has_single_class_label) {
    // single-class (one-hot) label for each vertex: N x 1
    using_single_class_labels_ = true;
    local_ground_truth_labels_.resize(partitioned_graph_->size());
  } else {
    // multi-class label for each vertex: N x num classes
    using_single_class_labels_ = false;
    local_ground_truth_labels_.resize(partitioned_graph_->size() *
                                      num_label_classes_);
  }

  size_t cur_gid              = 0;
  size_t found_local_vertices = 0;
  // each line contains a set of 0s and 1s
  std::string read_line;

  // loop through all labels of the graph
  while (std::getline(file_stream, read_line)) {
    // only process label if this node is local
    if (partitioned_graph_->isLocal(cur_gid)) {
      uint32_t cur_lid = partitioned_graph_->getLID(cur_gid);
      // read line as bitset of 0s and 1s
      std::istringstream label_stream(read_line);
      unsigned cur_bit;
      // bitset size is # of label classes
      for (size_t cur_class = 0; cur_class < num_label_classes_; ++cur_class) {
        // read a bit
        label_stream >> cur_bit;

        if (has_single_class_label) {
          // in single class, only 1 bit is set in bitset; that represents the
          // class to take
          if (cur_bit != 0) {
            // set class and break (assumption is that's the only bit that is
            // set)
            local_ground_truth_labels_[cur_lid] = cur_class;
            break;
          }
        } else {
          // else the entire bitset needs to be copied over to the label array
          // TODO this can possibly be saved all at once rather than bit by bit?
          local_ground_truth_labels_[cur_lid * num_label_classes_ + cur_class] =
              cur_bit;
        }
      }
      found_local_vertices++;
    }
    // always increment cur_gid
    cur_gid++;
  }

  file_stream.close();

  GALOIS_LOG_ASSERT(found_local_vertices == partitioned_graph_->size());
}

void galois::graphs::GNNGraph::ReadLocalFeatures(
    const std::string& dataset_name) {
  GALOIS_LOG_VERBOSE("[{}] Reading features from disk...", host_id_);

  // read in dimensions of features, specifically node feature length
  size_t num_global_vertices;

  std::string file_dims = input_directory_ + dataset_name + "-dims.txt";
  std::ifstream ifs;
  ifs.open(file_dims, std::ios::in);
  ifs >> num_global_vertices >> node_feature_length_;
  ifs.close();

  GALOIS_LOG_ASSERT(num_global_vertices == partitioned_graph_->globalSize());
  GALOIS_LOG_VERBOSE("[{}] N x D: {} x {}", host_id_, num_global_vertices,
                     node_feature_length_);

  // memory for all features of all nodes in graph
  // TODO read features without loading entire feature file into memory; this
  // is quite inefficient
  std::unique_ptr<GNNFloat[]> full_feature_set =
      std::make_unique<GNNFloat[]>(num_global_vertices * node_feature_length_);

  // read in all features
  std::ifstream file_stream;
  std::string feature_file = input_directory_ + dataset_name + "-feats.bin";
  file_stream.open(feature_file, std::ios::binary | std::ios::in);
  file_stream.read((char*)full_feature_set.get(), sizeof(GNNFloat) *
                                                      num_global_vertices *
                                                      node_feature_length_);
  file_stream.close();

  // allocate memory for local features
  local_node_features_.resize(partitioned_graph_->size() *
                              node_feature_length_);

  // copy over features for local nodes only
  size_t num_kept_vertices = 0;
  for (size_t gid = 0; gid < num_global_vertices; gid++) {
    if (partitioned_graph_->isLocal(gid)) {
      // copy over feature vector
      std::copy(full_feature_set.get() + gid * node_feature_length_,
                full_feature_set.get() + (gid + 1) * node_feature_length_,
                &local_node_features_[partitioned_graph_->getLID(gid) *
                                      node_feature_length_]);
      num_kept_vertices++;
    }
  }
  full_feature_set.reset();
  GALOIS_LOG_ASSERT(num_kept_vertices == partitioned_graph_->size());
}

//! Helper function to read masks from file into the appropriate structures
//! given a name, mask type, and arrays to save into
size_t galois::graphs::GNNGraph::ReadLocalMasksFromFile(
    const std::string& dataset_name, const std::string& mask_type,
    GNNRange* mask_range, char* masks) {
  size_t range_begin;
  size_t range_end;

  // read mask range
  std::string mask_filename =
      input_directory_ + dataset_name + "-" + mask_type + "_mask.txt";
  std::ifstream mask_stream;
  mask_stream.open(mask_filename, std::ios::in);
  mask_stream >> range_begin >> range_end >> std::ws;
  GALOIS_LOG_ASSERT(range_begin <= range_end);

  // set the range object
  mask_range->begin = range_begin;
  mask_range->end   = range_end;
  mask_range->size  = range_end - range_begin;

  size_t cur_line_num       = 0;
  size_t local_sample_count = 0;
  std::string line;
  // each line is a number signifying if mask is set for the vertex
  while (std::getline(mask_stream, line)) {
    std::istringstream mask_stream(line);
    // only examine vertices/lines in range
    if (cur_line_num >= range_begin && cur_line_num < range_end) {
      // only bother if node is local
      if (partitioned_graph_->isLocal(cur_line_num)) {
        unsigned mask = 0;
        mask_stream >> mask;
        if (mask == 1) {
          masks[partitioned_graph_->getLID(cur_line_num)] = 1;
          local_sample_count++;
        }
      }
    }
    cur_line_num++;
  }
  mask_stream.close();

  return local_sample_count;
}

void galois::graphs::GNNGraph::ReadLocalMasks(const std::string& dataset_name) {
  // allocate the memory for the local masks
  local_training_mask_.resize(partitioned_graph_->size());
  local_validation_mask_.resize(partitioned_graph_->size());
  local_testing_mask_.resize(partitioned_graph_->size());

  if (dataset_name == "reddit") {
    // TODO reddit is hardcode handled at the moment; better way to not do
    // this?
    global_training_mask_range_   = {.begin = 0, .end = 153431, .size = 153431};
    global_validation_mask_range_ = {
        .begin = 153431, .end = 153431 + 23831, .size = 23831};
    global_testing_mask_range_ = {
        .begin = 177262, .end = 177262 + 55703, .size = 55703};

    // training
    for (size_t i = global_training_mask_range_.begin;
         i < global_training_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_training_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }

    // validation
    for (size_t i = global_validation_mask_range_.begin;
         i < global_validation_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_validation_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }

    // testing
    for (size_t i = global_testing_mask_range_.begin;
         i < global_testing_mask_range_.end; i++) {
      if (partitioned_graph_->isLocal(i)) {
        local_testing_mask_[partitioned_graph_->getLID(i)] = 1;
      }
    }
  } else {
    // XXX i can get local sample counts from here if i need it
    ReadLocalMasksFromFile(dataset_name, "train", &global_training_mask_range_,
                           local_training_mask_.data());
    ReadLocalMasksFromFile(dataset_name, "val", &global_validation_mask_range_,
                           local_validation_mask_.data());
    ReadLocalMasksFromFile(dataset_name, "test", &global_testing_mask_range_,
                           local_testing_mask_.data());
  }
}

void galois::graphs::GNNGraph::ReadWholeGraph(const std::string& dataset_name) {
  std::string input_file = input_directory_ + dataset_name + ".csgr";
  GALOIS_LOG_VERBOSE("[{}] Reading entire graph: file to read is {}", host_id_,
                     input_file);
  galois::graphs::readGraph(whole_graph_, input_file);
}

void galois::graphs::GNNGraph::InitNormFactor() {
  GALOIS_LOG_VERBOSE("[{}] Initializing norm factors", host_id_);
  norm_factors_.resize(partitioned_graph_->size(), 0.0);

  // get the norm factor contribution for each node based on the GLOBAL graph
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->size()),
      [&](size_t local_id) {
        // translate lid into gid to get global degree
        size_t global_id     = partitioned_graph_->getGID(local_id);
        size_t global_degree = whole_graph_.edge_end(global_id) -
                               whole_graph_.edge_begin(global_id);
        // only set if non-zero
        if (global_degree != 0) {
          norm_factors_[local_id] =
              1.0 / std::sqrt(static_cast<float>(global_degree));
        }
      },
      galois::loopname("InitNormFactor"));
}

#ifdef GALOIS_ENABLE_GPU
void galois::graphs::GNNGraph::InitGPUMemory() {
  // create int casted CSR
  uint64_t* e_index_ptr = partitioned_graph_->row_start_ptr();
  uint32_t* e_dest_ptr  = partitioned_graph_->edge_dst_ptr();

  // + 1 because first element is 0 in BLAS CSRs
  std::vector<int> e_index(partitioned_graph_->size() + 1);
  std::vector<int> e_dest(partitioned_graph_->sizeEdges());

  // set in parallel
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->size() + 1),
      [&](size_t index) {
        if (index != 0) {
          if (e_index_ptr[index - 1] >
              static_cast<size_t>(std::numeric_limits<int>::max())) {
            GALOIS_LOG_FATAL("{} is too big a number for int arrays on GPUs",
                             e_index_ptr[index - 1]);
          }
          e_index[index] = static_cast<int>(e_index_ptr[index - 1]);
        } else {
          e_index[index] = 0;
        }
      },
      galois::loopname("GPUEdgeIndexConstruction"));
  galois::do_all(
      galois::iterate(static_cast<size_t>(0), partitioned_graph_->sizeEdges()),
      [&](size_t edge) {
        if (e_dest_ptr[edge] >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
          GALOIS_LOG_FATAL("{} is too big a number for int arrays on GPUs",
                           e_dest_ptr[edge]);
        }

        e_dest[edge] = static_cast<int>(e_dest_ptr[edge]);
      },
      galois::loopname("GPUEdgeDestConstruction"));

  gpu_memory_.SetGraphTopology(e_index, e_dest);
  e_index.clear();
  e_dest.clear();

  gpu_memory_.SetFeatures(local_node_features_, node_feature_length_);
  gpu_memory_.SetLabels(local_ground_truth_labels_);
  gpu_memory_.SetMasks(local_training_mask_, local_validation_mask_,
                       local_testing_mask_);
  gpu_memory_.SetNormFactors(norm_factors_);
}
#endif
