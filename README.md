# SEAL: Selective Alignment for SNNs

---

**Table of contents:**

- [Abstract](#abstract)
- [Dependency](#dependency)
- [Directory Tree](#directory)
- [Usage](#usage)

## Abstract

Spiking neural networks (SNNs), which are brain-inspired and spike-driven, achieve high energy efficiency. However, a performance gap between SNNs and artificial neural networks (ANNs) still remains. Knowledge distillation (KD) is commonly adopted to improve SNN performance, but existing methods typically enforce uniform alignment across all timesteps, either from a teacher network or through inter-temporal self-distillation, implicitly assuming equal reliability of per-timestep predictions. In practice, SNN predictions vary and evolve over time, and intermediate timesteps may be incorrect or temporally inconsistent even when the final aggregated output is correct. Under these conditions, applying uniform distillation can result in overly restrictive or misleading supervision, leading to conflicting gradient signals. To address this issue, we propose Selective Alignment Knowledge Distillation (SeAl-KD), which selectively aligns predictions from the teacher or other timesteps by equalizing correctly and incorrectly predicted logits at erroneous timesteps and reweighting temporal alignment based on confidence and inter-timestep similarity, thereby mitigating gradient conflicts and stabilizing SNN optimization. Extensive experiments on static image and neuromorphic event-based datasets demonstrate consistent improvements over existing distillation methods in performance.
## Dependency

The major dependencies of this repo are listed as below.

```
# Name                 Version
python                  3.10.14 
torch                   2.4.1
torchvision             0.19.1
tensorboard             2.17.1
spikingjelly            0.0.0.0.14
```

## Directory Tree

```
|-- data/                 # Datasets
|-- experiment/           # Experiment scripts
|   |-- cifar/            # CIFAR-10/100 experiments
|   |-- dvs/              # CIFAR10-DVS experiments
|   `-- imagenet/         # ImageNet experiments
|-- model/                # Model definitions
`-- util/                 # Utility functions
```

## Usage

1. **CIFAR-10/100 (ANN Training)**
    ```bash
    python experiment/cifar/ann.py --arch resnet19 --dataset CIFAR100 --train_batch_size 128 --val_batch_size 128 --data_path [data_path] --wd 5e-4 --num_epoch 300
    ```

2. **CIFAR-10/100 (SNN Distillation)**
    ```bash
    python experiment/cifar/main.py --stu_arch resnet18 --tea_arch resnet34 --dataset CIFAR10 --train_batch_size 128 --val_batch_size 128 --data_path [data_path] --tea_path [your tea_path] --wd 5e-4 --decay 0.5 --T 4 --num_epoch 300 --alpha 0.6 --beta 0.15 --lr 0.1
    ```

3. **CIFAR10-DVS (ANN Training)**
    ```bash
    python experiment/cifar/ann.py --dataset CIFAR10_DVS_SJ --data_path [data_path] --log_path [log_path] --arch resnet19 --num_epoch 300 --lr 0.2 --wd 5e-4 --T 10
    ```

4. **CIFAR10-DVS (SNN Distillation)**
    ```bash
    python experiment/dvs/main.py --stu_arch resnet18 --tea_arch resnet19 --dataset CIFAR10_DVS_SJ --data_path [data_path] --tea_path [your tea_path] --train_batch_size 32 --val_batch_size 32 --wd 5e-4 --decay 0.5 --T 10 --num_epoch 300 --alpha 0.6 --beta 0.15 --lr 0.1 
    ```

5. **ImageNet (ANN Training)**
    ```bash
    python experiment/imagenet/ann.py --arch resnet34 --dataset imagenet --data_path [data_path] --train_batch_size 512 --val_batch_size 512 --lr 0.1 --wd 1e-4 --num_epoch 100
    ```

6. **ImageNet (SNN Distillation)**
    ```bash
    python experiment/imagenet/main.py --stu_arch preact_resnet34 --tea_arch resnet34 --dataset imagenet --train_batch_size 512 --val_batch_size 512 --data_path [data_path] --tea_path [your tea_path] --wd 2e-5 --decay 0.2 --T 4 --num_epoch 100 --alpha 0.6 --beta 0.15 --lr 0.2
    ```

This project is derived from [snn_temporal_decoupling_distillation](https://github.com/Intelli-Chip-Lab/snn_temporal_decoupling_distillation).
Thanks for the original work and inspiration.