// Copyright (c) Vinnam Kim
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#ifndef __BIPARTITE_SOFT_MATCHING_CUH__
#define __BIPARTITE_SOFT_MATCHING_CUH__

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bipartite_soft_matching(
    const torch::Tensor &key_tensor,
    const int r,
    const bool class_token,
    const bool distill_token);

torch::Tensor compute_score(
    const torch::Tensor &key_tensor,
    const bool class_token,
    const bool distill_token);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> partition_groups(
    const torch::Tensor &score,
    const int r,
    const bool class_token,
    const bool distill_token);

#endif
