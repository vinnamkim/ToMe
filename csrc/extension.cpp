// Copyright (c) Vinnam Kim
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#include <torch/extension.h>
#include "merge.cuh"
#include "bipartite_soft_matching.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("merge_token_and_size", &merge_token_and_size,
          "Merge tokens and token sizes using CUDA.");
    m.def("bipartite_soft_matching", &bipartite_soft_matching,
          "Compute index groups (unmerged, source, destination) for merging with bipartite soft matching algorithm using CUDA.");
    m.def("compute_score", &compute_score,
          "Compute score matrix for bipartite matching using CUDA.");
    m.def("partition_groups", &partition_groups,
          "Partition index groups (unmerged, source, destination) using a C++ wrapper of the original Python Torch script.");
}
