# AUTOGENERATED! DO NOT EDIT! File to edit: 01_dataloader.ipynb (unless otherwise specified).

__all__ = ['MNIST_NORMALIZATION', 'AmbiguousMNIST', 'FastMNIST', 'DirtyMNIST']

# Cell

import os
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import torch
from torchvision.datasets.mnist import MNIST, VisionDataset
from torchvision.datasets.utils import download_url, extract_archive, verify_str_arg
from torchvision.transforms import Compose, Normalize, ToTensor

# based on torchvision.datasets.mnist.py (https://github.com/pytorch/vision/blob/37eb37a836fbc2c26197dfaf76d2a3f4f39f15df/torchvision/datasets/mnist.py)

MNIST_NORMALIZATION = Normalize((0.1307,), (0.3081,))

# Cell


class AmbiguousMNIST(VisionDataset):
    """
    Ambiguous-MNIST Dataset

    Please cite:

        @article{mukhoti2021deterministic,
          title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
          author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
          journal={arXiv preprint arXiv:2102.11582},
          year={2021}
        }


    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Whiten the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for max throughput)
    """

    mirrors = ["http://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v0.6.0/"]

    resources = dict(data=("amnist_samples.pt", None), targets=("amnist_labels.pt", None))

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        normalize: bool = True,
        device=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        self.data = torch.load(self.resource_path("data"), map_location=device)
        if normalize:
            self.data = self.data.sub_(0.1307).div_(0.3081)

        self.targets = torch.load(self.resource_path("targets"), map_location=device)

        num_multi_labels = self.targets.shape[1]

        self.data = self.data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1, 28, 28)
        self.targets = self.targets.reshape(-1)

        data_range = slice(None, 60000) if self.train else slice(60000, None)
        self.data = self.data[data_range]
        self.targets = self.targets[data_range]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def data_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    def resource_path(self, name):
        return os.path.join(self.data_folder, self.resources[name][0])

    def _check_exists(self) -> bool:
        return all(os.path.exists(self.resource_path(name)) for name in self.resources)

    def download(self) -> None:
        """Download the data if it doesn't exist in data_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.data_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources.values():
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_url(url, root=self.data_folder, filename=filename, md5=md5)
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                except:
                    raise
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

        print("Done!")

# Cell

# based on https://tinyurl.com/pytorch-fast-mnist
class FastMNIST(MNIST):
    """
    FastMNIST, like MNIST (<http://yann.lecun.com/exdb/mnist/>) but faster throughput.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Whiten the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for
            max throughput).
    """

    def __init__(self, *args, normalize, device, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

        if normalize:
            self.data = self.data.sub_(0.1307).div_(0.3081)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# Cell


def DirtyMNIST(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    normalize=True,
    device=None,
):
    """
    DirtyMNIST

    Please cite:

        @article{mukhoti2021deterministic,
          title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
          author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
          journal={arXiv preprint arXiv:2102.11582},
          year={2021}
        }

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Whiten the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for
            max throughput).
    """

    mnist_dataset = FastMNIST(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        normalize=normalize,
        device=device,
    )

    amnist_dataset = AmbiguousMNIST(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        normalize=normalize,
        device=device,
    )

    return torch.utils.data.ConcatDataset([mnist_dataset, amnist_dataset])