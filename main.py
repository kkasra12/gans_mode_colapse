import gc
import json
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
from evaluate import DATAFILE


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
        shared_layers: int = 0,
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
        self.data_file = DATAFILE
        self.checkpoint_dir = None
        self.run_id = None
        self.shared_layers = shared_layers
        self.ngpu = ngpu

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
        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )
        # self.device = torch.device("cuda:0")
        self.nz = nz
        self.netG, self.netD = create_models(
            ngpu=ngpu,
            nlabels=len(self.dataset.all_classes),
            nz=nz,
            ngf=ngf,
            nc=nc,
            ndf=ndf,
            shared_layers=shared_layers,
            device=self.device,
        )
        self.criterion = nn.BCELoss()
        print(self.device)

        self.netG.to(self.device)
        self.netD.to(self.device)

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
        run_id: int = -1,
    ):
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.current_data_file = os.path.join(checkpoint_dir, f"{self.data_file}.json")

        last_run_id = self.find_last_run_id()
        if run_id == -1:
            self.run_id = last_run_id + 1
        elif run_id <= last_run_id:
            self.run_id = run_id
        else:
            raise ValueError(
                f"run_id should be less than or equal to {last_run_id}, got {run_id},"
                f"for more info, please check the {os.path.join(checkpoint_dir, f'{self.data_file}.json')} file."
            )

        self["num_epochs"] = num_epochs
        self["ngpu"] = self.ngpu
        self["nlabels"] = len(self.dataset.all_classes)
        self["generate_images_per_epoch"] = generate_images_per_epoch
        self["batch_size"] = self.batch_size
        self["shared_layers"] = self.shared_layers
        self["nz"] = self.nz
        self["ngf"] = self.netG.ngf
        self["nc"] = self.netG.nc
        self["ndf"] = self.netD.ndf
        self["lr"] = self.optimizerD.param_groups[0]["lr"]
        self["beta1"] = self.optimizerD.param_groups[0]["betas"][0]
        self["device"] = self.device.type

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
                    gc.collect()
                    # torch.cuda.empty_cache()
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
        self.checkpoint_dir = None

    def __setitem__(self, key, value):
        """
        we will save a file with name of "{self.data_file}.json" in the checkpoint_dir,
        the keys are "run_{self.run_id}" and the values are the key-value pairs passed to this function.
        """
        if self.checkpoint_dir is None:
            return
        new_value = {key: value}
        with open(os.path.join(self.current_data_file), "r+") as f:
            current_values = json.load(f)
            the_key = f"run_{self.run_id}"
            assert isinstance(current_values, dict), f"Bad data: {current_values}"
            assert (
                the_key in current_values
            ), f"We can't find {the_key} in the data file ({self.current_data_file})."
            current_values[the_key].update(new_value)
            f.seek(0)
            json.dump(current_values, f)

    def __getitem__(self, key):
        if self.checkpoint_dir is None:
            return
        with open(os.path.join(self.current_data_file), "r") as f:
            return json.load(f)[f"run_{self.run_id}"].get(key)

    def find_last_run_id(self, create_dict: bool = True):
        """
        Finds the last run ID from the data file and optionally creates a new dictionary entry.

        Args:
            create_dict (bool, optional): Whether to create a new dictionary entry. Defaults to True.

        Returns:
            int: The last run ID found in the data file.
        """
        with open(self.current_data_file, "a+") as f:
            if (f_read := f.read()) != "":
                data = json.load(f_read)
                # note that the keys are in the format of "run_{id}"
                last_id = max(map(lambda x: x.split("_")[1], data.keys()))
            else:
                data = {}
                last_id = 0

        if create_dict:
            with open(self.current_data_file, "w") as f:
                data[f"run_{last_id + 1}"] = {}
                json.dump(data, f)
            return last_id

    def save_checkpoint(self, epoch, checkpoint_dir, vars: dict = None):
        suffix = f"{epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        torch.save(
            self.netG.state_dict(),
            os.path.join(checkpoint_dir, (g_name := f"netG_{suffix}.pth")),
        )
        torch.save(
            self.netD.state_dict(),
            os.path.join(checkpoint_dir, (d_name := f"netD_{suffix}.pth")),
        )

        if vars:
            with open(
                os.path.join(checkpoint_dir, (v_name := f"vars_{suffix}.pkl")), "wb"
            ) as f:
                pickle.dump(vars, f)

        g_files = self["g_files"]
        if g_files:
            g_files.append(g_name)
        else:
            g_files = [g_name]
        self["g_files"] = g_files

        d_files = self["d_files"]
        if d_files:
            d_files.append(d_name)
        else:
            d_files = [d_name]
        self["d_files"] = d_files

        v_files = self["v_files"]
        if v_files:
            v_files.append(v_name)
        else:
            v_files = [v_name]
        self["v_files"] = v_files

    def load_checkpoint(self, checkpoint_dir, epoch):
        raise NotImplementedError


if __name__ == "__main__":
    fire.Fire(Main)
    # for instance run with very default parameters:
    # python main.py train
