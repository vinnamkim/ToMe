// Copyright (c) Vinnam Kim
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#include <c10/cuda/CUDAStream.h>

#include "merge.cuh"
#include "helper.cuh"

template <typename U>
__global__ void copy_cuda_kernel(
    const U *__restrict__ input,
    const U *__restrict__ size,
    const int64_t *__restrict__ unm_idx,
    const int64_t *__restrict__ src_idx,
    const int64_t *__restrict__ dst_idx,
    U *__restrict__ output,
    U *__restrict__ new_size,
    const int B,
    const int L,
    const int C,
    const int L_out,
    const int L_unm,
    const int r)
{
    const int linear_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (linear_idx >= B * L_out * C)
        return;

    const int b = linear_idx / (L_out * C);
    const int b_mod = linear_idx - b * L_out * C;
    const int l_out = b_mod / C;
    const int c = b_mod - l_out * C;

    const int l_in = (l_out < L_unm)
                         ? 2 * static_cast<int>(unm_idx[b * L_unm + l_out])
                         : 2 * (l_out - L_unm) + 1;

    float v_output = static_cast<float>(size[b * L + l_in]) * static_cast<float>(input[b * L * C + l_in * C + c]);
    float v_new_size = static_cast<float>(size[b * L + l_in]);

    // This linear search algorithm is cheaper than reduce_scatter for a small r
    for (int i = 0; i < r; i++)
    {
        const int l_dst = 2 * static_cast<int>(dst_idx[b * r + i]) + 1;
        if (l_dst != l_in)
            continue;

        const int l_src = 2 * static_cast<int>(src_idx[b * r + i]);

        const float tmp_new_size = v_new_size + static_cast<float>(size[b * L + l_src]);
        const float tmp_output = (v_new_size / tmp_new_size) * v_output;

        v_output = tmp_output +
                   static_cast<float>(size[b * L + l_src]) /
                       tmp_new_size *
                       static_cast<float>(input[b * L * C + l_src * C + c]);
        v_new_size = tmp_new_size;
    }

    if (c == 0)
        new_size[b * L_out + l_out] = static_cast<U>(v_new_size);

    output[b * L_out * C + l_out * C + c] = static_cast<U>(v_output);
}

std::tuple<torch::Tensor, torch::Tensor> merge_token_and_size(
    const torch::Tensor &input,
    const torch::Tensor &size,
    const torch::Tensor &unm_idx,
    const torch::Tensor &src_idx,
    const torch::Tensor &dst_idx,
    const bool distill_token)
{
    CHECK_TENSOR(input, 3);
    CHECK_TENSOR(size, 3);
    CHECK_TENSOR(unm_idx, 3);
    CHECK_TENSOR(src_idx, 3);
    CHECK_TENSOR(dst_idx, 3);

    CHECK_SHAPE_EQ(input, size, 2);

    const int64_t r = src_idx.size(1);

    if (r == 0)
        return {input, size};

    const int64_t B = input.size(0);
    const int64_t L = input.size(1);
    const int64_t C = input.size(2);

    const int64_t L_unm = unm_idx.size(1);
    const int64_t L_out = L - r;

    const auto cuda_stream = at::cuda::getCurrentCUDAStream(input.device().index());

    auto output = torch::empty({B, L_out, C}, input.options());
    auto new_size = torch::empty({B, L_out, 1}, input.options());

    dim3 threadsPerBlock{512};
    dim3 blocksPerGrid{(B * L_out * C + threadsPerBlock.x - 1) / threadsPerBlock.x};

    if (input.scalar_type() == c10::ScalarType::Float)
    {
        copy_cuda_kernel<float><<<blocksPerGrid, threadsPerBlock, 0, cuda_stream.stream()>>>(
            reinterpret_cast<const float *>(input.const_data_ptr()),
            reinterpret_cast<const float *>(size.const_data_ptr()),
            reinterpret_cast<const int64_t *>(unm_idx.const_data_ptr()),
            reinterpret_cast<const int64_t *>(src_idx.const_data_ptr()),
            reinterpret_cast<const int64_t *>(dst_idx.const_data_ptr()),
            reinterpret_cast<float *>(output.data_ptr()),
            reinterpret_cast<float *>(new_size.data_ptr()),
            B, L, C, L_out, L_unm, r);
    }
    else if (input.scalar_type() == c10::ScalarType::Half)
    {
        copy_cuda_kernel<at::Half><<<blocksPerGrid, threadsPerBlock, 0, cuda_stream.stream()>>>(
            reinterpret_cast<const at::Half *>(input.const_data_ptr()),
            reinterpret_cast<const at::Half *>(size.const_data_ptr()),
            reinterpret_cast<const int64_t *>(unm_idx.const_data_ptr()),
            reinterpret_cast<const int64_t *>(src_idx.const_data_ptr()),
            reinterpret_cast<const int64_t *>(dst_idx.const_data_ptr()),
            reinterpret_cast<at::Half *>(output.data_ptr()),
            reinterpret_cast<at::Half *>(new_size.data_ptr()),
            B, L, C, L_out, L_unm, r);
    }
    else
    {
        TORCH_CHECK(false, "Not support data type=", c10::toString(input.scalar_type()));
    }

    // TODO: Remove this and enhance the logic to cover an arbitrary number of tokens to be protected.
    if (distill_token)
    {
        auto relocate_distill_token = [&](torch::Tensor &x)
        {
            return torch::cat(
                {
                    x.index({"...",
                             torch::indexing::Slice(torch::indexing::None, 1),
                             torch::indexing::Slice()}),
                    x.index({"...",
                             torch::indexing::Slice(L_unm, L_unm + 1),
                             torch::indexing::Slice()}),
                    x.index({"...",
                             torch::indexing::Slice(1, L_unm),
                             torch::indexing::Slice()}),
                    x.index({"...",
                             torch::indexing::Slice(L_unm + 1, torch::indexing::None),
                             torch::indexing::Slice()}),
                },
                -2);
        };
        output = relocate_distill_token(output);
        new_size = relocate_distill_token(new_size);
    }

    return {output, new_size};
}
