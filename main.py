import gc
import os
from datetime import datetime
import pickle
from typing import Optional

from tqdm import tqdm
from torchvision.transforms import v2 as transforms
from torch import nn, optim
import torch


from data import MnistDataset
from evaluate import DATAFILE, Evaluate
from logger import Logger
from model import create_models


class Main:
    logger: Logger
    device: torch.device

    def __init__(
        self,
        data_path: str = "./data",
        batch_size: int = 128,
        ngpu: int = 1,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 1,
        ndf: int = 64,
        lr: float = 0.0002,
        transform: bool = True,
        beta1: float = 0.5,
        shared_layers: int = 0,
        device: Optional[str] = None,
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
            device (str): The device to use for training. if None, it will be set to "cuda:0" if a GPU is available, otherwise "cpu". Default is None.
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
            transform_funcs = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.RandomRotation(degrees=(-20, 20)),
                    transforms.Resize(size=(64, 64)),
                    transforms.RandomCrop(size=(50, 50)),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.Resize(size=(64, 64)),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset = MnistDataset(
            data_path, batch_size=batch_size, transform=transform_funcs
        )
        if device is None:
            self.device = torch.device(
                "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
            )
        else:
            self.device = torch.device(device)
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
        print("Using device:", self.device)

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
        continue_training: bool = False,
        use_wandb: bool = True,
    ):
        """
        Trains the GAN model for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model (default: 5).
            checkpoint_dir (str | os.PathLike): The directory to save checkpoints (default: "./checkpoints").
            generate_images_per_epoch (int): The number of images to generate per epoch (default: 10).
            run_id (int): The ID of the current run (default: -1).
            continue_training (bool): Whether to continue training from the "run_id" (default: False).
            use_wandb (bool): Whether to use Weights & Biases for logging (default: True).

        Raises:
            ValueError: If the specified run_id is greater than the last run ID found in the checkpoint directory.

        Returns:
            None
        """

        logger = Logger(
            log_folder=checkpoint_dir,
            run_id=run_id,
            resume=continue_training,
            use_wandb=use_wandb,
        )

        if not continue_training:
            # last_run_id = self.find_last_run_id()
            # if run_id == -1:
            #     self.run_id = last_run_id + 1
            #     # if run_id is -1, then we should generate a new id
            # elif run_id <= last_run_id:
            #     self.run_id = run_id
            # else:
            #     raise ValueError(
            #         f"run_id should be less than or equal to {last_run_id}, got {run_id},"
            #         f"for more info, please check the {os.path.join(checkpoint_dir, f'{self.data_file}.json')} file."
            #     )

            # self["num_epochs"] = num_epochs
            # self["ngpu"] = self.ngpu
            # self["nlabels"] = len(self.dataset.all_classes)
            # self["generate_images_per_epoch"] = generate_images_per_epoch
            # self["batch_size"] = self.batch_size
            # self["shared_layers"] = self.shared_layers
            # self["nz"] = self.nz
            # self["ngf"] = self.netG.ngf
            # self["nc"] = self.netG.nc
            # self["ndf"] = self.netD.ndf
            # self["lr"] = self.optimizerD.param_groups[0]["lr"]
            # self["beta1"] = self.optimizerD.param_groups[0]["betas"][0]
            # self["device"] = self.device.type
            # print(f"Starting training for run_id {self.run_id}...")

            logger["num_epochs"] = num_epochs
            logger["ngpu"] = self.ngpu
            logger["nlabels"] = len(self.dataset.all_classes)
            logger["generate_images_per_epoch"] = generate_images_per_epoch
            logger["batch_size"] = self.batch_size
            logger["shared_layers"] = self.shared_layers
            logger["nz"] = self.nz
            logger["ngf"] = self.netG.ngf
            logger["nc"] = self.netG.nc
            logger["ndf"] = self.netD.ndf
            logger["lr"] = self.optimizerD.param_groups[0]["lr"]
            logger["beta1"] = self.optimizerD.param_groups[0]["betas"][0]
            logger["device"] = self.device.type

        else:
            # if we want to continue training, we should load the last model and optimizer states
            # if run_id == -1:
            #     run_id = self.find_last_run_id(create_dict=False)
            # data_json = self.load_run_json(run_id)
            # if data_json is None:
            #     raise ValueError(f"Can't find the run_id {run_id} in the data file.")
            # self.run_id = run_id

            assert (
                (currenct_epoch := len(logger.get_files("g_files")))
                == len(logger.get_files("d_files"))
                == len(logger.get_files("v_files"))
            ), "The number of files should be the same."

            num_epochs = logger["num_epochs"] - currenct_epoch
            if num_epochs == 0:
                raise ValueError(
                    f"Training for {run_id} has finished with {logger.get('num_epochs')} epochs."
                )

            # check if all hyperparameters are the same
            assert logger.get("ngpu") == self.ngpu
            assert logger.get("nlabels") == len(self.dataset.all_classes)
            assert logger.get("generate_images_per_epoch") == generate_images_per_epoch
            assert logger.get("batch_size") == self.batch_size
            assert logger.get("shared_layers") == self.shared_layers
            assert logger.get("nz") == self.nz
            assert logger.get("ngf") == self.netG.ngf
            assert logger.get("nc") == self.netG.nc
            assert logger.get("ndf") == self.netD.ndf
            assert logger.get("lr") == self.optimizerD.param_groups[0]["lr"]
            assert logger.get("beta1") == self.optimizerD.param_groups[0]["betas"][0]
            assert logger.get("device") == self.device.type

            # load the model
            self.netG.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoint_dir, logger.get_files("g_files")[-1].file_path
                    )
                )
            )
            self.netD.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoint_dir, logger.get_files("d_files")[-1].file_path
                    )
                )
            )
            self.optimizerG.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoint_dir, logger.get_files("g_files")[-1].file_path
                    )
                )
            )
            self.optimizerD.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoint_dir, logger.get_files("d_files")[-1].file_path
                    )
                )
            )

        real_label = 1.0
        fake_label = 0.0

        # after each `step` batches, generate some images using `fixed_noise` and save them in the `imgs` list
        # fixed_noise = torch.randn(
        #     generate_images_per_epoch, self.nz, 1, 1, device=self.device
        # )
        fixed_noise = self.netG.make_sample_input(self.batch_size, device=self.device)

        step = len(self.dataset) // generate_images_per_epoch
        all_labels = torch.tensor(sorted(self.dataset.all_classes), device=self.device)

        imgs = []
        g_losses = []
        d_losses = []

        for epoch in range(num_epochs):
            retry = 5
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
                    # noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                    fake = self.netG(
                        self.netG.make_sample_input(b_size, device=self.device), lbl
                    )

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
                    logger.log("g_loss", errG.item())
                    logger.log("d_loss", errD.item())
                    # end of one batch
                except Exception as e:
                    print(f"Exception occured in batch {i}, {e}")
                    gc.collect()
                    if (retry := retry - 1) == 0:
                        raise e
                    # torch.cuda.empty_cache()
                # end of one epoch
                # Check how the generator is doing by saving G's output on fixed_noise
                if i % step == 0:
                    # start of the step
                    with torch.no_grad():
                        fake = self.netG(fixed_noise, all_labels).detach().cpu()
                    imgs.append(fake)
                    logger.log("fid", Evaluate.fid(device=self.device, netG=self.netG))
                    is_mean, is_std = Evaluate.inception_score(
                        device=self.device, netG=self.netG
                    )
                    logger.log("is_mean", is_mean.item())
                    logger.log("is_std", is_std.item())

            if checkpoint_dir:
                self.save_checkpoint(
                    epoch,
                    checkpoint_dir,
                    logger,
                    vars={"g_losses": g_losses, "d_losses": d_losses, "imgs": imgs},
                )
        self.checkpoint_dir = None

    # def __setitem__(self, key, value):
    #     """
    #     we will save a file with name of "{self.data_file}.json" in the checkpoint_dir,
    #     the keys are "run_{self.run_id}" and the values are the key-value pairs passed to this function.
    #     """
    #     if self.checkpoint_dir is None:
    #         return
    #     new_value = {key: value}
    #     with open(os.path.join(self.current_data_file), "r+") as f:
    #         current_values = json.load(f)
    #         the_key = f"run_{self.run_id}"
    #         assert isinstance(current_values, dict), f"Bad data: {current_values}"
    #         assert (
    #             the_key in current_values
    #         ), f"We can't find {the_key} in the data file ({self.current_data_file})."
    #         current_values[the_key].update(new_value)
    #         f.seek(0)
    #         json.dump(current_values, f)

    # def __getitem__(self, key):
    #     if self.checkpoint_dir is None:
    #         return
    #     with open(os.path.join(self.current_data_file), "r") as f:
    #         return json.load(f)[f"run_{self.run_id}"].get(key)

    # def find_last_run_id(self, create_dict: bool = True):
    #     """
    #     Finds the last run ID from the data file and optionally creates a new dictionary entry.
    #     If no data file(self.current_data_file) exists, it will return 0.

    #     Args:
    #         create_dict (bool, optional): Whether to create a new dictionary entry. Defaults to True.

    #     Returns:
    #         int: The last run ID found in the data file.
    #     """
    #     if not os.path.exists(self.current_data_file):
    #         print(f"Creating a new data file: {self.current_data_file=}")
    #         data = {}
    #         last_id = 0
    #     else:
    #         with open(self.current_data_file, "r") as f:
    #             data = json.load(f)
    #             # note that the keys are in the format of "run_{id}"
    #             last_id = int(max(map(lambda x: int(x.split("_")[1]), data.keys())))

    #     if create_dict:
    #         with open(self.current_data_file, "w") as f:
    #             data[f"run_{last_id + 1}"] = {}
    #             json.dump(data, f)
    #     return last_id

    def save_checkpoint(
        self, epoch, checkpoint_dir, logger: Logger, vars: Optional[dict] = None
    ):
        suffix = f"{epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        torch.save(
            self.netG.state_dict(),
            (g_name := os.path.join(checkpoint_dir, f"netG_{suffix}.pth")),
        )
        print(f"Saving {g_name}")

        # g_files = self["g_files"]
        # if g_files:
        #     g_files.append(g_name)
        # else:
        #     g_files = [g_name]
        # self["g_files"] = g_files
        logger.log_file(
            file_path=g_name,
            category="g_files",
            metadata={"epoch": epoch},
            description="Generator model",
        )

        torch.save(
            self.netD.state_dict(),
            (d_name := os.path.join(checkpoint_dir, f"netD_{suffix}.pth")),
        )
        print(f"Saving {d_name}")

        # d_files = self["d_files"]
        # if d_files:
        #     d_files.append(d_name)
        # else:
        #     d_files = [d_name]
        # self["d_files"] = d_files
        logger.log_file(
            file_path=d_name,
            category="d_files",
            metadata={"epoch": epoch},
            description="Discriminator model",
        )

        if vars:
            with open(
                (v_name := os.path.join(checkpoint_dir, f"vars_{suffix}.pkl")), "wb"
            ) as f:
                pickle.dump(vars, f)
            print(f"Saving {v_name}")

            # v_files = self["v_files"]
            # if v_files:
            #     v_files.append(v_name)
            # else:
            #     v_files = [v_name]
            # self["v_files"] = v_files
            logger.log_file(
                file_path=v_name,
                category="v_files",
                metadata={"epoch": epoch},
                description="Variables",
            )

    # def load_checkpoint(self, checkpoint_dir, epoch):
    #     raise NotImplementedError

    # def load_run_json(self, run_id):
    #     data_file = os.path.join(self.checkpoint_dir, f"{self.data_file}.json")
    #     with open(data_file, "r") as f:
    #         return json.load(f)[f"run_{run_id}"]


if __name__ == "__main__":
    Main(lr=0.0001).train(num_epochs=3)
