import gc
import os
import pickle
from datetime import datetime

import fire
import torch
from torch import nn, optim
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from data import MnistDataset
from model import create_models


class Main:
    def __init__(
        self,
        data_path: str = "./data",
        batch_size: int = 12,
        ngpu: int = 1,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 1,
        ndf: int = 64,
        lr: float = 0.0002,
        transform: bool = True,
        beta1: float = 0.5,
        workers: int = 1,
    ):
        """
        Initializes the GANsModeCollapse class.

        Args:
            data_path (str): The path to the data directory. Default is "./data".
            batch_size (int): The batch size for training. Default is 64.
            ngpu (int): The number of GPUs to use. Default is 1.
            nz (int): The size of the input noise vector. Default is 100.
            ngf (int): The number of filters in the generator. Default is 64.
            nc (int): The number of channels in the input image. Default is 1.
            ndf (int): The number of filters in the discriminator. Default is 64.
            lr (float): The learning rate for the Adam optimizer. Default is 0.0002.
            transform (bool): Whether to apply data transformations. Default is True.
            beta1 (float): The beta1 parameter for the Adam optimizer. Default is 0.5.
            workers (int): The number of worker threads for data loading. Default is 1.
        """
        # TODO: Add id as the argument to save the model with the id in the filename
        # Dont allow to run the same id again, if the id exists, raise an error, if id==-1 then generate a new id
        # TODO: Add a json file to save each run parameters for each id
        # TODO: Change the output format to use the id in the filename
        if transform:
            # TODO: Try to use transforms.RandomApply
            transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomCrop(size=(20, 20)),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.RandomPerspective(),
                    transforms.RandomZoomOut(),
                    transforms.Resize(size=(64, 64)),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset = MnistDataset(
            data_path, batch_size=batch_size, transform=transform
        )

        self.nz = nz
        self.netG, self.netD = create_models(
            ngpu=ngpu,
            nlabels=len(self.dataset.all_classes),
            nz=nz,
            ngf=ngf,
            nc=nc,
            ndf=ndf,
        )
        self.criterion = nn.BCELoss()

        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=lr, betas=(beta1, 0.999)
        )

    def train(
        self,
        num_epochs: int = 5,
        checkpoint_dir: str | os.PathLike = "./checkpoints",
        generate_images_per_epoch: int = 10,
    ):
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        real_label = 1.0
        fake_label = 0.0

        # after each `step` batches, generate some images using `fixed_noise` and save them in the `imgs` list
        fixed_noise = torch.randn(
            generate_images_per_epoch, self.nz, 1, 1, device=self.device
        )
        step = len(self.dataset) // generate_images_per_epoch
        all_labels = torch.tensor(sorted(self.dataset.all_classes), device=self.device)

        imgs = []
        g_losses = []
        d_losses = []

        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, (data, lbl) in enumerate(tqdm(self.dataset)):
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    ## Train with all-real batch
                    self.netD.zero_grad()
                    # Format batch
                    real_cpu = data.to(self.device)
                    b_size = real_cpu.size(0)
                    assert (
                        b_size == self.batch_size
                    ), f"Batch size is {b_size} but expected {self.batch_size}"
                    label = torch.full(
                        (b_size,), real_label, dtype=torch.float, device=self.device
                    )
                    output = self.netD(real_cpu).view(-1)
                    errD_real = self.criterion(output, label)
                    errD_real.backward()
                    # D_x = output.mean().item()
                    # can be used for further analysis (if needed)

                    ## Train with all-fake batch
                    noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                    fake = self.netG(noise, lbl)

                    label.fill_(fake_label)
                    # Classify all fake batch with Discriminator
                    output = self.netD(fake.detach()).view(-1)
                    # Calculate Discriminator's loss on the all-fake batch
                    errD_fake = self.criterion(output, label)
                    errD_fake.backward()
                    # D_G_z1 = output.mean().item()
                    # can be used for further analysis (if needed)
                    errD = errD_real + errD_fake
                    self.optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    self.netG.zero_grad()
                    # fake labels are real for generator cost
                    label.fill_(real_label)
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = self.netD(fake).view(-1)
                    # Calculate G's loss based on this output
                    errG = self.criterion(output, label)
                    errG.backward()
                    # D_G_z2 = output.mean().item()
                    # can be used for further analysis (if needed)
                    self.optimizerG.step()

                    # Save Losses for plotting later
                    g_losses.append(errG.item())
                    d_losses.append(errD.item())
                except Exception as e:
                    print(f"Exception occured in batch {i}, {e}")
                # Check how the generator is doing by saving G's output on fixed_noise
                if i % step == 0:
                    with torch.no_grad():
                        fake = self.netG(fixed_noise, all_labels).detach().cpu()
                    imgs.append(fake)

            if checkpoint_dir:
                self.save_checkpoint(
                    epoch,
                    checkpoint_dir,
                    vars={"g_losses": g_losses, "d_losses": d_losses, "imgs": imgs},
                )

    def save_checkpoint(self, epoch, checkpoint_dir, vars: dict = None):
        suffix = f"{epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        torch.save(
            self.netG.state_dict(), os.path.join(checkpoint_dir, f"netG_{suffix}.pth")
        )
        torch.save(
            self.netD.state_dict(), os.path.join(checkpoint_dir, f"netD_{suffix}.pth")
        )
        if vars:
            with open(os.path.join(checkpoint_dir, f"vars_{suffix}.pkl"), "wb") as f:
                pickle.dump(vars, f)


if __name__ == "__main__":
    fire.Fire(Main)
    # for instance run with very default parameters:
    # python main.py train
