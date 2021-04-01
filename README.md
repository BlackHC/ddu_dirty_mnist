# DDU's Dirty-MNIST
> You'll never want to use MNIST again for OOD or AL.


## Install

`pip install ddu_dirty_mnist`

## How to use

After installing, you get a Dirty-MNIST train or test set just like you would for MNIST in PyTorch.

```python
# gpu

import ddu_dirty_mnist

dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST(
    ".", train=True, download=True, device="cuda"
)
dirty_mnist_test = ddu_dirty_mnist.DirtyMNIST(
    ".", train=False, download=True, device="cuda"
)
len(dirty_mnist_train), len(dirty_mnist_test)
```




    (120000, 30000)



Here is how to create `torch.utils.data.DataLoader`, see [the documentation](./dataloader.html) for details.

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

If you only care about Ambiguous-MNIST, you can use:

```python
# gpu

import ddu_dirty_mnist

ambiguous_mnist_train = ddu_dirty_mnist.AmbiguousMNIST(
    ".", train=True, download=True, device="cuda"
)
ambiguous_mnist_test = ddu_dirty_mnist.AmbiguousMNIST(
    ".", train=False, download=True, device="cuda"
)

ambiguous_mnist_train, ambiguous_mnist_test
```




    (Dataset AmbiguousMNIST
         Number of datapoints: 60000
         Root location: .,
     Dataset AmbiguousMNIST
         Number of datapoints: 20000
         Root location: .)



Here is how to create `torch.utils.data.DataLoader`, see [the documentation](./dataloader.html) for details.

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
