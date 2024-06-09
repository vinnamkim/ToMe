// Copyright (c) Vinnam Kim
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#ifndef __MERGE_CUH__
#define __MERGE_CUH__

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> merge_token_and_size(
    const torch::Tensor &input,
    const torch::Tensor &size,
    const torch::Tensor &unm_idx,
    const torch::Tensor &src_idx,
    const torch::Tensor &dst_idx,
    const bool distill_token);

#endif
