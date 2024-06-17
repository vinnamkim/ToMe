# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tome_cuda_extension",
    version="0.1",
    author="Vinnam Kim",
    author_email="vinnam.kim@gmail.com",
    url="https://github.com/vinnamkim/ToMe-CUDA-Extension",
    description="CUDA extension for Token Merging for Vision Transformers",
    install_requires=[
        "torchvision",
        "numpy",
        "timm==0.4.12",
        "pillow",
        "tqdm",
        "scipy",
    ],
    packages=find_packages(exclude=("examples", "build")),
    ext_modules=[
        CUDAExtension(
            name="tome_cuda",
            sources=[
                "csrc/extension.cpp",
                "csrc/merge.cu",
                "csrc/bipartite_soft_matching.cu",
            ],
            include_dirs=["cutlass/include"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
