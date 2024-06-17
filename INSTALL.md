# Installation

## Requirements
```bash
 - python >= 3.8
 - pytorch >= 1.12.1  # For scatter_reduce
 - torchvision        # With matching version for your pytorch install
 - timm == 0.4.12     # Might work on other versions, but this is what we tested
 - jupyter            # For example notebooks
 - scipy              # For visualization and sometimes torchvision requires it
```

Your system also should be available to compile CUDA source files.
Please visit https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc for details.
We tested our code under CUDA 12.1.

## Setup
First, clone the repository:
```bash
git clone https://github.com/vinnamkim/ToMe-CUDA-Extension
cd ToMe-CUDA-Extension
git submodule update --init  # Pull CUTLASS
```
Then set up the `ToMe-CUDA-Extension` package with:
```bash
pip install .
```
