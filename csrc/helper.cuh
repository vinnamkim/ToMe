// Copyright (c) Vinnam Kim
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#ifndef __HELPER_CUH__
#define __HELPER_CUH__

#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " should be on a CUDA device.");
#define CHECK_DIM(x, n_dim) TORCH_CHECK(x.dim() == n_dim, #x " should have ", std::to_string(n_dim), " dimensions.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " should be contiguous.");
#define CHECK_TENSOR(x, n_dim) \
    CHECK_CUDA(x);             \
    CHECK_DIM(x, n_dim);       \
    CHECK_CONTIGUOUS(x);

#define CHECK_SHAPE_EQ(a, b, n_dim)                                                         \
    {                                                                                       \
        for (int i = 0; i < n_dim; i++)                                                     \
        {                                                                                   \
            TORCH_CHECK(a.size(i) == b.size(i), #a " and " #b " should have equal shapes.") \
        }                                                                                   \
    }

#endif
