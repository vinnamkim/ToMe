# PyTorch CUDA extension for Token Merging: Your ViT but Faster

This repository includes a PyTorch CUDA extension for [Token Merging: Your ViT but Faster](https://arxiv.org/abs/2210.09461). Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman.
We have enhanced the performance of [the original codebase](https://github.com/facebookresearch/ToMe), which was implemented in pure PyTorch Python code, by introducing a PyTorch CUDA extension.

Our improvements mainly focus on reducing overheads such as:

1. Explicitly splitting a key tensor into two tensors to compute distance metric scores for soft bipartite matching.
2. Using PyTorch expand and scatter reduce operations to merge two tensors into one according to the matched indices.

Our method offers the following enhancements:

1. Exploits the GemmArray API of [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass): This allows for implicit computation of distance metric scores without explicitly splitting the tensor into two.
2. Implements custom CUDA kernels: These custom kernels avoid using PyTorch expand and scatter reduce operations, which are typically too costly.

You can see the actual performance improvement in [this notebook](examples/1_benchmark_timm.ipynb).
We also attached the summary table here.

1. Experimental setup

    | Model                | Input size         | Precision | GPU                         |
    |----------------------|--------------------|-----------|-----------------------------|
    | vit_base_patch16_224 | [256, 3, 224, 224] | FP16      | NVIDIA® GeForce RTX™ 4060Ti |

2. Results with `r=8`

    |                        | No ToMe | Pure PyTorch ToMe | CUDA extension ToMe | Additional Gain |
    |------------------------|---------|-------------------|---------------------|-----------------|
    | Throughput (im/s)      | 522.04  | 580.03            | 635.16              | 55.13           |
    | Throughput improvement | 1       | 1.11              | 1.22                | 0.11            |

3. Results with `r=(8, -1.0)`

    |                        | No ToMe | Pure PyTorch ToMe | CUDA extension ToMe | Additional Gain |
    |------------------------|---------|-------------------|---------------------|-----------------|
    | Throughput (im/s)      | 522.04  | 660.27            | 717.80              | 57.53           |
    | Throughput improvement | 1       | 1.26              | 1.37                | 0.11            |

4. Results with `r=16`

    |                        | No ToMe | Pure PyTorch ToMe | CUDA extension ToMe | Additional Gain |
    |------------------------|---------|-------------------|---------------------|-----------------|
    | Throughput (im/s)      | 522.04  | 868.17            | 941.25              | 73.08           |
    | Throughput improvement | 1       | 1.66              | 1.80                | 0.14            |

5. Results with `r=(16, -1.0)`

    |                        | No ToMe | Pure PyTorch ToMe | CUDA extension ToMe | Additional Gain |
    |------------------------|---------|-------------------|---------------------|-----------------|
    | Throughput (im/s)      | 522.04  | 1304.62           | 1395.00             | 90.38           |
    | Throughput improvement | 1       | 2.50              | 2.67                | 0.17            |

> NOTE: The 4th column represents the absolute difference between the 3rd and 2nd columns.

## Installation
See [INSTALL.md](INSTALL.md) for installation details.

## License and Contributing

This work is licensed under the [CC-BY-NC 4.0](LICENSE).

## Original Repository

- https://github.com/facebookresearch/ToMe

## Customizations

- Implement a PyTorch CUDA extension to improve the performance of original method.
