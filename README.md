# DDU's Dirty-MNIST
> You'll never want to use MNIST again for OOD or AL.


[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2102.11582-B31B1B.svg)](https://arxiv.org/abs/2102.11582)
[![Pytorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/BlackHC/ddu_dirty_mnist/blob/master/LICENSE)

This repository contains the code for [*Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty*](https://arxiv.org/abs/2102.11582).

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{mukhoti2021deterministic,
  title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
  author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
  journal={arXiv preprint arXiv:2102.11582},
  year={2021}
}
```
---

## Install

`pip install ddu_dirty_mnist`

## How to use

After installing, you get a Dirty-MNIST train or test set just like you would for MNIST in PyTorch.

```python
# gpu

import ddu_dirty_mnist

dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST(".", train=True, download=True, device="cuda")
dirty_mnist_test = ddu_dirty_mnist.DirtyMNIST(".", train=False, download=True, device="cuda")
len(dirty_mnist_train), len(dirty_mnist_test)
```




    (120000, 70000)



Create `torch.utils.data.DataLoader`s with `num_workers=0, pin_memory=False` for maximum throughput, see [the documentation](01_dataloader.ipynb) for details.

```python
# gpu
import torch

dirty_mnist_train_dataloader = torch.utils.data.DataLoader(
    dirty_mnist_train,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)
dirty_mnist_test_dataloader = torch.utils.data.DataLoader(
    dirty_mnist_test,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
```

### Ambiguous-MNIST

If you only care about Ambiguous-MNIST, you can use:

```python
# gpu

import ddu_dirty_mnist

ambiguous_mnist_train = ddu_dirty_mnist.AmbiguousMNIST(".", train=True, download=True, device="cuda")
ambiguous_mnist_test = ddu_dirty_mnist.AmbiguousMNIST(".", train=False, download=True, device="cuda")

ambiguous_mnist_train, ambiguous_mnist_test
```




    (Dataset AmbiguousMNIST
         Number of datapoints: 60000
         Root location: .,
     Dataset AmbiguousMNIST
         Number of datapoints: 60000
         Root location: .)



Again, create `torch.utils.data.DataLoader`s with `num_workers=0, pin_memory=False` for maximum throughput, see [the documentation](./dataloader.html) for details.

```python
# gpu
import torch

ambiguous_mnist_train_dataloader = torch.utils.data.DataLoader(
    ambiguous_mnist_train,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)
ambiguous_mnist_test_dataloader = torch.utils.data.DataLoader(
    ambiguous_mnist_test,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
```

## Additional Guidance

1. DirtyMNIST is a concatenation of MNIST + AmbiguousMNIST, with 60k samples each in the training set.
1. The current AmbiguousMNIST contains 6k unique samples with 10 labels each. This multi-label dataset gets flattened to 60k samples. The assumption is that amibguous samples have multiple "valid" labels as they are ambiguous. MNIST samples are intentionally undersampled (in comparison), which benefits AL acquisition functions that can select unambiguous samples.
1. Pick your initial training samples (for warm starting Active Learning) from the MNIST half of DirtyMNIST to avoid starting training with potentially very ambiguous samples, which might add a lot of variance to your experiments.
1. Make sure to pick your validation set from the MNIST half as well, for the same reason as above.
1. Make sure that your batch acquisition size is >= 10 (probably) given that there are 10 multi-labels per samples in Ambiguous-MNIST.
1. By default, Gaussian noise with stddev 0.05 is added to each sample to prevent acquisition functions from cheating by disgarding "duplicates".
1. If you want to split Ambiguous-MNIST into subsets (or Dirty-MNIST within the second ambiguous half), make sure to split by multiples of 10 to avoid splits within a flattened multi-label sample.
