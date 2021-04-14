# DDU's Dirty-MNIST
> You'll never want to use MNIST again for OOD or AL.


---
You can find the paper here: https://arxiv.org/abs/2102.11582.

Please cite us using:

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

    Downloading http://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v0.6.0/amnist_samples.pt
    Using downloaded and verified file: ./AmbiguousMNIST/amnist_samples.pt
    
    Downloading http://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v0.6.0/amnist_labels.pt
    Downloading https://github-releases.githubusercontent.com/351788366/3f7f7380-9be7-11eb-8a0e-e58bb5dfc2bb?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210414%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210414T093659Z&X-Amz-Expires=300&X-Amz-Signature=22b376aa098be2c61beef6a3ed03a6fb32ab2b9dec31776b04ac321c13ffbbcc&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=351788366&response-content-disposition=attachment%3B%20filename%3Damnist_labels.pt&response-content-type=application%2Foctet-stream to ./AmbiguousMNIST/amnist_labels.pt
    
    Done!





    (120000, 70000)



Create `torch.utils.data.DataLoader`s with `num_workers=0, pin_memory=False` for maximum throughput, see [the documentation](./dataloader.html) for details.

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
