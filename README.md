# DAEBM one dimension experiment

本代码旨在实现**Persistent Trained, Diffusion-assisted Energy-based Models**中的一维EBM实验以及一维DAEBM实验。其中一维EBM实验主要修改自[开源仓库](https://github.com/XinweiZhang/DAEBM)中的一维Gauss代码（R语言）。DAEBM实验主要参考论文中二维高斯环装实验的网络设置，其他参考MNIST网络实现。

主要内容为按照论文中设置的能量函数进行一维EBM实验，以及自行设计能量函数进行EBM实验，基于扩散辅助的能量模型的一维DAEBM实验。

## Requirements

本实验在`python3.11`和`torch2.1.2`下可以成功运行，其他环境要求`python`至少在`3.10`及以上（代码中需要用到该版本及以后的相关特性）。其他安装的库可见`requirements.txt`（为可以运行的库，其他版本不做保证）

``` shell
pip install -r requirements.txt
```

## Train & Test

`./src`文件夹包含了训练文件，包括：

* `oneD_Gaussian_DAEBM.py`：一维高斯DAEBM实验
* `oneD_Gaussian.py`：一维高斯EBM实验
* `post_sampling.py`：只包含DAEBM的post sampling实验，EBM的post sampling在训练结束后直接运行。

运行代码时，可以运行`scripts/script_oneDGaussian.sh`。下面一个运行示例：

``` shell
bash scripts/script_oneDGaussian.sh --program_type oneD_Gaussian_DAEBM
```
