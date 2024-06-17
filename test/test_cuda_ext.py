import pytest
import torch
from tome_cuda import (
    bipartite_soft_matching,
    compute_score,
    partition_groups,
)
from tome.merge import (
    bipartite_soft_matching_cuda,
    bipartite_soft_matching,
    merge_wavg,
    compute_score_torch,
    partition_groups_torch,
)


class TestCudaExt:

    @pytest.fixture(params=range(0, 5))
    def fxt_seed(self, request):
        seed = request.param
        torch.manual_seed(seed)

    @pytest.fixture(
        params=[(1, 197, 256), (64, 64, 64), (1, 16, 16)], ids=["p1", "p2", "p3"]
    )
    def fxt_input(self, fxt_seed, request):
        shape = request.param
        return torch.randint(-4, 5, shape, device="cuda").float()

    @pytest.mark.parametrize(
        "dtype", [torch.float, torch.float16], ids=["float", "half"]
    )
    @pytest.mark.parametrize("distill_token", [False, True])
    @pytest.mark.parametrize("class_token", [True, False])
    @pytest.mark.parametrize("r", [8, 16])
    def test_bipartite_soft_matching(
        self,
        fxt_input: torch.Tensor,
        r: int,
        class_token: bool,
        distill_token: bool,
        dtype: torch.dtype,
    ) -> None:
        fxt_input = fxt_input.to(dtype=dtype)
        size = torch.ones(
            [fxt_input.shape[0], fxt_input.shape[1], 1],
            device=fxt_input.device,
            dtype=fxt_input.dtype,
        )

        merge_cuda_ext, _ = bipartite_soft_matching_cuda(
            fxt_input.clone(),
            r,
            class_token,
            distill_token,
        )

        merge, _ = bipartite_soft_matching(
            fxt_input.clone(),
            r,
            class_token,
            distill_token,
        )

        actual_merged, actual_size = merge_cuda_ext(fxt_input, size)
        expect_merged, expect_size = merge_wavg(merge, fxt_input, size)

        assert torch.allclose(actual_merged, expect_merged, atol=1e-3)
        assert torch.allclose(actual_size, expect_size, atol=1e-3)

    @pytest.mark.parametrize(
        "dtype", [torch.float, torch.float16], ids=["float", "half"]
    )
    @pytest.mark.parametrize("distill_token", [False, True])
    @pytest.mark.parametrize("class_token", [True, False])
    def test_compute_score(
        self,
        fxt_input: torch.Tensor,
        class_token: bool,
        distill_token: bool,
        dtype: torch.dtype,
    ) -> None:
        fxt_input = fxt_input.to(dtype=dtype)

        actual = compute_score(fxt_input.clone(), class_token, distill_token)
        expect = compute_score_torch(fxt_input.clone(), class_token, distill_token)

        assert torch.allclose(actual, expect, atol=1e-3)

    @pytest.mark.parametrize(
        "dtype", [torch.float, torch.float16], ids=["float", "half"]
    )
    @pytest.mark.parametrize("distill_token", [False, True])
    @pytest.mark.parametrize("class_token", [True, False])
    @pytest.mark.parametrize("r", [8, 16])
    def test_partition_groups(
        self,
        fxt_input: torch.Tensor,
        r: int,
        class_token: bool,
        distill_token: bool,
        dtype: torch.dtype,
    ) -> None:
        fxt_input = fxt_input.to(dtype=dtype)

        scores = compute_score(fxt_input, class_token, distill_token)

        actual = partition_groups(scores, r, class_token, distill_token)
        expect = partition_groups_torch(scores, r, class_token)

        for a, e in zip(actual, expect):
            assert torch.allclose(a, e)
