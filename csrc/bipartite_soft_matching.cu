// Copyright (c) Vinnam Kim
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <c10/cuda/CUDAStream.h>

#include "bipartite_soft_matching.cuh"
#include "helper.cuh"

template <typename U>
cudaError_t cutlass_array_sgemm(
    int batch_size,
    int m,
    int n,
    int k,
    U const *const *A,
    int lda,
    U const *const *B,
    int ldb,
    U *const *C,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream)
{
    using ElementA = U;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = U;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = U;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementAccumulator = float;

    using Gemm = cutlass::gemm::device::GemmArray<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator>;

    Gemm gemm_op;

    cutlass::Status status = gemm_op(
        {{m, n, k},
         A,
         lda,
         B,
         ldb,
         C,
         ldc,
         C,
         ldc,
         {alpha, beta},
         batch_size},
        nullptr,
        stream);

    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

    return cudaSuccess;
}

template <typename U>
void get_score_cuda_impl(
    const U *input,
    U *score,
    cudaStream_t stream,
    const int batch_size,
    const int length,
    const int m,
    const int n,
    const int k,
    const bool class_token,
    const bool distill_token)
{
    const size_t batch_stride_T = static_cast<size_t>(length * k);
    const size_t batch_stride_C = static_cast<size_t>(m * n);

    std::vector<const U *> ptr_vec(3 * batch_size);

    for (size_t b_idx = 0; b_idx < batch_size; b_idx++)
    {
        ptr_vec[b_idx] = input + b_idx * batch_stride_T;                  // Start from row = 0
        ptr_vec[b_idx + batch_size] = input + b_idx * batch_stride_T + k; // Skip row = 0 and start from row = 1
        ptr_vec[b_idx + 2 * batch_size] = score + b_idx * batch_stride_C;
    }

    U **ptr;
    cudaError_t result;

    result = cudaMallocAsync(&ptr, 3 * batch_size * sizeof(U *), stream);
    TORCH_CHECK(result == cudaSuccess, "cudaMalloc failed.");

    result = cudaMemcpyAsync(ptr, ptr_vec.data(), 3 * batch_size * sizeof(U *), cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(result == cudaSuccess, "cudaMemcpy failed.");

    const U *const *ptr_A = reinterpret_cast<const U *const *>(ptr);
    const U *const *ptr_B = reinterpret_cast<const U *const *>(ptr + batch_size);
    U *const *ptr_C = ptr + 2 * batch_size;

    TORCH_CHECK(
        cutlass_array_sgemm<U>(batch_size, m, n, k,
                               ptr_A, 2 * k,
                               ptr_B, 2 * k,
                               ptr_C, n,
                               1.0f, 0.0f, stream) == cudaSuccess,
        "Failed to launch cutlass array sgemm kernel");

    result = cudaFreeAsync(ptr, stream);
    TORCH_CHECK(result == cudaSuccess, "cudaFree failed.");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bipartite_soft_matching(
    const torch::Tensor &key_tensor,
    const int r,
    const bool class_token,
    const bool distill_token)
{
    const auto score = compute_score(key_tensor, class_token, distill_token);
    return partition_groups(score, r, class_token, distill_token);
}

torch::Tensor compute_score(
    const torch::Tensor &key_tensor,
    const bool class_token,
    const bool distill_token)
{
    CHECK_TENSOR(key_tensor, 3);

    const auto norm = key_tensor.norm(2, -1, true);
    key_tensor.div_(norm);

    const int batch_size = key_tensor.size(0);
    const int length = key_tensor.size(1);
    const int k = key_tensor.size(2);

    const int m = length / 2 + length % 2;
    const int n = length / 2;

    auto score = torch::empty({batch_size, m, n}, key_tensor.options());

    const auto cuda_stream = at::cuda::getCurrentCUDAStream(key_tensor.device().index());

    if (key_tensor.scalar_type() == c10::ScalarType::Float)
    {
        get_score_cuda_impl<float>(
            reinterpret_cast<const float *>(key_tensor.const_data_ptr()),
            reinterpret_cast<float *>(score.data_ptr()),
            cuda_stream.stream(),
            batch_size,
            length,
            m,
            n,
            k,
            class_token, distill_token);
    }
    else if (key_tensor.scalar_type() == c10::ScalarType::Half)
    {
        get_score_cuda_impl<cutlass::half_t>(
            reinterpret_cast<const cutlass::half_t *>(key_tensor.data_ptr()),
            reinterpret_cast<cutlass::half_t *>(score.data_ptr()),
            cuda_stream.stream(),
            batch_size,
            length,
            m,
            n,
            k,
            class_token, distill_token);
    }
    else
    {
        TORCH_CHECK(false, "Not support data type=", c10::toString(key_tensor.scalar_type()));
    }

    if (class_token)
        score.index_put_({"...", 0, torch::indexing::Slice()}, -10000.0f);
    if (distill_token)
        score.index_put_({"...", torch::indexing::Slice(), 0}, -10000.0f);

    return score;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> partition_groups(
    const torch::Tensor &score,
    const int r,
    const bool class_token,
    const bool distill_token)
{
    const auto batch_size = score.size(0);
    const auto [node_max, node_idx] = score.max(-1, false);
    const auto edge_idx = node_max.argsort(-1, true);
    edge_idx.unsqueeze_(-1);

    const auto unm_idx = edge_idx.index(
                                     {"...",
                                      torch::indexing::Slice(r, torch::indexing::None),
                                      torch::indexing::Slice()})
                             .contiguous();
    const auto src_idx = edge_idx.index(
                                     {"...",
                                      torch::indexing::Slice(torch::indexing::None, r),
                                      torch::indexing::Slice()})
                             .contiguous();
    const auto dst_idx = node_idx.index(
                                     {"...", torch::indexing::None})
                             .gather(-2, src_idx);

    if (class_token)
    {
        const auto [sorted_unm_idx, _] = unm_idx.sort(1, false);
        return {sorted_unm_idx, src_idx, dst_idx};
    }

    return {unm_idx, src_idx, dst_idx};
}
