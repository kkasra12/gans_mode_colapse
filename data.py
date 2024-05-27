import gzip
import os
import struct
import urllib.request

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset
from torchvision import datasets


class MnistDataset(IterableDataset):
    def __init__(
        self, data: os.PathLike, batch_size: int, nc=1, transform=None, shuffle=True
    ):
        """
        Initialize the MnistDataset object.

        Args:
            data (os.PathLike): The path to the data directory.
            batch_size (int): The batch size for data loading.
            nc (int, optional): Number of channels in the images. (the images in the __next__ output will have shape of [H, W, nc]) Defaults to 1.
            transform (optional): Transformations to apply to the images. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        super(MnistDataset).__init__()
        self.nc = nc
        self.data = data = os.path.join(data, "MNIST", "raw")
        if not os.path.isdir(data):
            self.download(data)
        self.images, self.labels = self.load_data()
        self.transform = transform
        self.batch_size = batch_size
        if shuffle:
            idx = np.arange(len(self.images))
            np.random.shuffle(idx)
            self.images: np.ndarray = self.images[idx]
            self.labels: np.ndarray = self.labels[idx]
        self.all_classes = np.unique(self.labels)
        self.data_accumalator: dict[int, list[np.ndarray]] = {
            c: [] for c in self.all_classes
        }
        # this dictionary keys are the classes, and values are the list of images of that class.
        # well you can see the exact algorithm in the `__iter__` and `__next__` methods, but im
        # going to spoil it for you.
        # the idea is to iterate over the images (i.e. take 2*self.batchsize data) and store them into a dictionary with
        # the class as the key, and the list of images of that class as the value.
        # to be more specific about the steps:
        # in each __next__ call, we take one image and label, and store them into the dictionary with corresponding class.
        # choose one of the classes (keys) which have more than self.batchsize image in value and return the value of that key.

    def download(self, data: os.PathLike):
        os.makedirs(data, exist_ok=True)
        # TODO: use these two hosts to download the data, use the other one if the first one fails
        # hosts = [
        #     "http://yann.lecun.com/exdb/mnist/",
        #     "https://ossci-datasets.s3.amazonaws.com/mnist/",
        # ]
        # TODO: also, use these MD5 hashes to check the integrity of the downloaded files
        # resources = [
        #     ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        #     ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        #     ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        #     ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
        # ]

        urls = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        ]
        try:
            for url in urls:
                filename = os.path.join(data, os.path.basename(url))
                if not os.path.exists(filename):
                    print(f"Downloading {url} to {filename}")
                    urllib.request.urlretrieve(url, filename)
                if not os.path.exists(filename[:-3]):
                    with gzip.open(filename, "rb") as f_in:
                        with open(filename[:-3], "wb") as f_out:
                            f_out.write(f_in.read())
        except Exception as e:
            print(f"Failed to download the data: {e}, using pytorch functions")
            datasets.MNIST(data, download=True)

    def load_data(self):
        with open(os.path.join(self.data, "train-images-idx3-ubyte"), "rb") as f:
            _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read the image data
            images0 = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_images, rows, cols
            )

        with open(os.path.join(self.data, "t10k-images-idx3-ubyte"), "rb") as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read the image data
            images1 = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_images, rows, cols
            )

        images = np.concatenate((images0, images1))[..., None].repeat(self.nc, axis=-1)

        assert images.shape[1:] == (
            28,
            28,
            self.nc,
        ), f"Bad shape: {images.shape}, expected (70000, 28, 28)"
        assert images.dtype == np.uint8, f"Bad dtype: {images.dtype}, expected np.uint8"

        with open(os.path.join(self.data, "train-labels-idx1-ubyte"), "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            labels0 = np.frombuffer(f.read(), dtype=np.uint8)

        with open(os.path.join(self.data, "t10k-labels-idx1-ubyte"), "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            labels1 = np.frombuffer(f.read(), dtype=np.uint8)

        labels = np.concatenate((labels0, labels1))
        assert (
            labels.shape[0] == images.shape[0]
        ), f"Bad shape: {labels.shape}, expected {images.shape[0]}"
        assert labels.dtype == np.uint8, f"Bad dtype: {labels.dtype}, expected np.uint8"

        return images, labels

    def __iter__(self):
        self.data_accumalator = {c: [] for c in self.all_classes}
        self.idx = 0
        return self

    def __next__(self):
        while max(map(len, self.data_accumalator.values())) < self.batch_size:
            if self.idx >= len(self.images):
                # TODO: this should not be ended here,
                raise StopIteration
            img = self.images[self.idx]
            label = self.labels[self.idx]
            self.idx += 1
            # self.data_accumalator.setdefault(label, []).append(img)
            self.data_accumalator[label].append(img)
        most_frequent_class = max(
            self.data_accumalator, key=lambda k: len(self.data_accumalator[k])
        )
        batched_imgs = self.data_accumalator[most_frequent_class]
        self.data_accumalator[most_frequent_class] = []
        assert (
            len(batched_imgs) == self.batch_size
        ), f"Expected {self.batch_size} images, got {len(batched_imgs)}, {batched_imgs[0].shape}"

        if self.transform:
            batched_imgs = [self.transform(img) for img in batched_imgs]
        else:
            batched_imgs = [torch.tensor(img) for img in batched_imgs]
        batched_imgs = torch.stack(batched_imgs)

        return batched_imgs, most_frequent_class

    def __len__(self):
        return len(self.labels) // self.batch_size


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MnistDataset("data", batch_size=12)
    for i, (imgs, label) in enumerate(DataLoader(dataset)):
        print(f"Batch {i} has {len(imgs)} images of class {label}")
        plt.imshow(np.concatenate(imgs, axis=1).squeeze(), cmap="gray")
        plt.title(f"Batch {i} has {len(imgs)} images of class {label}")
        plt.show()
        if i == 3:
            break
