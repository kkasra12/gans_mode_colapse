from cProfile import label
import copy
import warnings
from matplotlib.style import available
from numpy import iterable
import torch
from torch import nn

__doc__ = """
A DCGAN is a direct extension of the GAN, except that it
explicitly uses convolutional and convolutional-transpose layers in the
discriminator and generator, respectively. It was first described by
Radford et. al. in the paper [Unsupervised Representation Learning With
Deep Convolutional Generative Adversarial
Networks](https://arxiv.org/pdf/1511.06434.pdf). The discriminator is
made up of strided
[convolution](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)
layers, [batch
norm](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d)
layers, and
[LeakyReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU)
activations. The input is a 3x64x64 input image and the output is a
scalar probability that the input is from the real data distribution.
The generator is comprised of
[convolutional-transpose](https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d)
layers, batch norm layers, and
[ReLU](https://pytorch.org/docs/stable/nn.html#relu) activations. The
input is a latent vector, $z$, that is drawn from a standard normal
distribution and the output is a 3x64x64 RGB image. The strided
conv-transpose layers allow the latent vector to be transformed into a
volume with the same shape as an image. In the paper, the authors also
give some tips about how to setup the optimizers, how to calculate the
loss functions, and how to initialize the model weights, all of which
will be explained in the coming sections.
"""


def weights_init(m):
    """
    Initialize the weights of a module.
    From the DCGAN paper, the authors specify that all model weights shall
    be randomly initialized from a Normal distribution with `mean=0`,
    `stdev=0.02`. The `weights_init` function takes an initialized model as
    input and reinitializes all convolutional, convolutional-transpose, and
    batch normalization layers to meet this criteria. This function is
    applied to the models immediately after initialization.

    Args:
        m (nn.Module): The module to initialize the weights for.

    Returns:
        None
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(
        self,
        ngpu: int,
        number_of_generators: int,
        shared_layers: int,
        nz: int,
        ngf: int,
        nc: int,
    ):
        """
        Initializes the Generator class.

        Args:
            ngpu (int): Number of GPUs to use.
            number_of_generators (int): Number of classes for conditional generation.
            number_shared_layers (int, optional): Number of shared layers (not implemented yet).
            nz (int, optional): Size of the input noise vector.
            ngf (int, optional): Number of generator filters in the first layer.
            nc (int, optional): Number of channels in the output image.

            the tested values are:
                number_shared_layers: int = 0,
                nz: int = 100,
                ngf: int = 64,
                nc: int = 1,
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.number_of_generators = number_of_generators
        self.shared_layers = shared_layers
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        available_layers = [
            [
                # input is Z, going into a convolution
                nn.ConvTranspose2d(
                    in_channels=nz,
                    out_channels=ngf * 8,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ],
            [  # state size. ``(ngf*8) x 4 x 4``
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.Dropout(0.2),
            ],
            [  # state size. ``(ngf*4) x 8 x 8``
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.Dropout(0.3),
            ],
            [  # state size. ``(ngf*2) x 16 x 16``
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.Dropout(0.4),
            ],
            [  # state size. ``(ngf) x 32 x 32``
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
                # # state size. ``(nc) x 64 x 64`]
                # nn.ReLU(),
            ],
        ]
        if shared_layers >= len(available_layers):
            raise ValueError(
                f"shared_layers should be less than {len(available_layers)} but got {shared_layers}"
            )

        tmp = []
        for _ in range(shared_layers):
            tmp.extend(available_layers.pop(0))
        self.shared_layers = nn.Sequential(*tmp)

        tmp = []
        # we need to flatten the available_layers list
        while available_layers:
            tmp.extend(available_layers.pop(0))

        self.models: list[nn.Sequential] = [
            copy.deepcopy(nn.Sequential(*tmp)) for _ in range(number_of_generators)
        ]

        for i, model in enumerate(self.models):
            self.add_module(f"model_{i}", model)
            # print(f"model_{i}", model, "-----", sep="\n")

    def forward(self, inputs, labels: torch.Tensor):
        """
        :param inputs: The input tensor
        :param labels: The labels tensor (they must be some integers)
        """
        if iterable(labels):
            # t = [
            #     self.models[lbl.item()](inp.unsqueeze(0)).squeeze(0)
            #     for lbl, inp in zip(labels, inputs, strict=True)
            # ]
            # raise NotImplementedError(
            #     "labels should be a tensor, iters are not supported yet"
            # )
            t = []
            if len(labels) != len(inputs):
                warnings.warn(
                    f"The length of labels and inputs are not equal, {len(labels)=} != {len(inputs)=}",
                    stacklevel=3,
                )
            for lbl, inp in zip(labels, inputs):
                shared_layers_out = self.shared_layers(inp.unsqueeze(0))
                t.append(self.models[lbl.item()](shared_layers_out).squeeze(0))
            return torch.stack(t)
        return self.models[labels.item()](self.shared_layers(inputs))

    def generate(self, num_samples: int, labels: torch.Tensor = None):
        """
        Generate samples from the generator.

        Args:
            num_samples (int): Number of samples to generate.
            labels (torch.Tensor, optional): The labels tensor. if None, it will be randomly generated. Defaults to None.

        Returns:
            torch.Tensor: The generated samples.
        """
        if labels is None:
            labels = torch.randint(0, self.number_of_generators, (num_samples,))
        imgs = [self(torch.randn(1, self.nz), lbl) for lbl in labels]
        return torch.stack(imgs)

    def make_sample_input(self, num_samples: int, device: torch.device | str = None):
        """
        Make a sample input for the generator.
        for instance you can say:
        ```
        labels = ...
        assert len(labels) == 5 # for instance
        gen = Generator(...)
        gen(gen.make_sample_input(5), labels)
        ```
        it will generate 5 random images.
        """
        if device:
            return torch.randn(num_samples, self.nz, 1, 1, device=device)
        return torch.randn(num_samples, self.nz, 1, 1)

    def make_sample_labels(self, num_samples: int, device: torch.device | str = None):
        """
        Make a sample labels for the generator.
        for instance you can say:
        ```
        labels = netG.make_sample_labels(5)
        assert len(labels) == 5
        gen = Generator(...)
        gen(gen.make_sample_input(5), labels)
        ```
        it will generate 5 random images.
        """
        if device:
            return torch.randint(
                0, self.number_of_generators, (num_samples,), device=device
            )
        return torch.randint(0, self.number_of_generators, (num_samples,))


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        """
        Initializes the Discriminator class.

        Args:
            ngpu (int): Number of GPUs to use.
            nc (int, optional): Number of input channels. Defaults to 1.
            ndf (int, optional): Number of discriminator filters. Defaults to 64.
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        out = self.main(input)
        return out


def create_models(
    ngpu: int,
    nlabels: int,
    device: torch.device | str,
    nz: int,
    ngf: int,
    nc: int,
    ndf: int,
    shared_layers: int,
    verbose: bool = False,
):
    """
    Create the generator and discriminator models for GAN training.

    Args:
        ngpu (int): Number of GPUs available for training.
        nlabels (int): Number of labels for conditional GAN.
        device (torch.device | str, optional): Device to use for training. If None, it will be automatically selected based on GPU availability. Defaults to None.
        number_shared_layers (int, optional): Number of shared layers between generator and discriminator. Defaults to 0.
        nz (int, optional): Size of the input noise vector. Defaults to 100.
        ngf (int, optional): Number of generator filters. Defaults to 64.
        nc (int, optional): Number of channels. Defaults to 1. used as the output of the generator and the input of the discriminator.
        ndf (int, optional): Number of discriminator filters. Defaults to 64.
        verbose (bool, optional): Whether to print the model architectures. Defaults to False.

    Returns:
        Generator, Discriminator: The generator and discriminator models.
    """
    if ngpu > 1:
        raise NotImplementedError("Multi-GPU is not implemented yet")
    if device is None:
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )
    elif isinstance(device, str):
        device = torch.device(device)

    netG = Generator(
        ngpu,
        number_of_generators=nlabels,
        shared_layers=shared_layers,
        nz=nz,
        ngf=ngf,
        nc=nc,
    ).to(device)

    # Handle multi-GPU if desired
    # if (device.type == "cuda") and (ngpu > 1):
    #     netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu, nc=nc, ndf=ndf).to(device)

    # Handle multi-GPU if desired
    # if (device.type == "cuda") and (ngpu > 1):
    #     netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    if verbose:
        print(netG)
        print(netD)

    return netG, netD


if __name__ == "__main__":
    # this is just a test
    netG, netD = create_models(
        nc=1,
        ngf=64,
        ndf=64,
        nlabels=(nlabels := 10),
        nz=(nz := 100),
        ngpu=0,
        shared_layers=3,
        device="cpu",
        verbose=True,
    )
    print("netG", netG, "-----", sep="\n")
    print("netD", netD, "-----", sep="\n")

    print(
        (
            fake_imgs := netG(
                torch.randn((batch_size := 3), nz, 1, 1),
                torch.randint(0, nlabels, (batch_size,)),
            )
        )
    )
    print(netD(fake_imgs))
