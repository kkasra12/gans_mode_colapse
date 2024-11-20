import json
import os
import pickle

# import fire
from tqdm import trange
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.nn.functional import interpolate as torch_interpolate

from data import MnistDataset
from model import Generator, create_models
from logger import Logger

# TODO: This variable should not be hardcoded.
DATAFILE = "data"


class Evaluate:
    @staticmethod
    def __initiate_dataset(batch_size: int):
        return MnistDataset(DATAFILE, batch_size=batch_size)

    @staticmethod
    def load_generator(logger: Logger):
        """
        Load the generator model from the logger.

        Args:
            logger (Logger): The logger object to load the generator model from.

        Returns:
            The generator model.
        """
        ngpu = logger["ngpu"]
        nlabels = logger["nlabels"]
        device = logger["device"]
        nz = logger["nz"]
        nc = logger["nc"]
        ngf = logger["ngf"]
        ndf = logger["ndf"]
        shared_layers = logger["shared_layers"]

        netG, _ = create_models(
            ngpu=ngpu,
            nlabels=nlabels,
            device=device,
            nz=nz,
            nc=nc,
            ngf=ngf,
            ndf=ndf,
            shared_layers=shared_layers,
        )
        netG.load_state_dict(
            torch.load(logger.get_files("g_files")[-1].file_path, weights_only=True)
        )
        netG.eval()
        return netG, device

    def show_imgs(self, run_id, prefix: str = None):
        """
        Display the generated images from the saved checkpoint.

        Args:
            run_id (int, optional): The ID of the model to load and generate images from. Defaults to None.
            prefix (str, optional): The prefix of the saved image files. if None, it will show the images instead of saving them. Defaults to None.

        Raises:
            ValueError: If the images are not saved in the checkpoint.

        Returns:
            None
        """
        raise RuntimeError("This function is not implemented yet.")
        print(f"Loading the data file {DATAFILE}.json...")
        with open(os.path.join(self.checkpoint_dir, f"{DATAFILE}.json")) as f:
            run_data = json.load(f).get(f"run_{run_id}")
            if run_data is None:
                raise ValueError("The run_id is not found in the data file.")
            v_file = run_data["v_files"][-1]

        # this v_file is the last epoch variables file containing the images, g_loss, d_loss.
        with open(os.path.join(self.checkpoint_dir, v_file), "rb") as f:
            saved_vars = pickle.load(f)

        if "imgs" not in saved_vars:
            raise ValueError(
                f"The images are not saved in the checkpoint, the saved vars are: {saved_vars.keys()}"
            )

        imgs = np.array(saved_vars["imgs"])
        print(imgs.shape)
        t = []
        for epoch, img in enumerate(
            imgs.reshape(imgs.shape[0], -1, imgs.shape[-1]), start=1
        ):
            break
            t.append(img)
            if len(t) == 10:
                img = np.concatenate(t, axis=1)
                plt.imshow(img, cmap="gray")
                plt.title(f"Generated images from run_{run_id} epoch {epoch} ")
                plt.axis("off")

                if prefix:
                    plt.savefig(
                        f"{prefix}_{run_id}_{epoch}_generated.png", bbox_inches="tight"
                    )
                    plt.clf()
                else:
                    plt.show()
                t = []

        # also, lets plot the losses
        print(len(saved_vars["g_losses"]), len(saved_vars["d_losses"]))
        plt.plot(saved_vars["g_losses"][:15], label="Generator Loss")
        plt.plot(saved_vars["d_losses"][:15], label="Discriminator Loss")
        plt.xlabel("Epoch")
        # change the xticks to be from 0 to 15
        # plt.xticks(np.arange(16))
        plt.ylabel("Loss")
        plt.legend()
        if prefix:
            plt.savefig(f"{prefix}_{run_id}_loss.png")
        else:
            plt.show()

    @classmethod
    def fid(
        cls,
        num_epochs: int = 100,
        batch_size: int = 10,
        *,
        device: torch.device | str = None,
        logger: Logger = None,
        netG: Generator = None,
    ) -> float:
        """
        Calculate the FID score for the generated images.

        Args:
            num_epochs (int, optional): The number of epochs to generate the images. Defaults to 100.
            batch_size (int, optional): The batch size to use for the FID calculation.
                                        (Therefor, final number of used image will be num_epochs * batch_size)s. Defaults to 10.
            device (torch.device|str, optional): The device to use for the calculation.
                                                  if you are using the logger (netG==None),
                                                  it will be loaded from the logger and this will be ignored. if None, it will use GPU if available. Defaults to None.
            logger (Logger, optional): The logger object to load the generator model from. if None, you should provide the netG.
            netG (Generator, optional): The generator model to use for the calculation. if None, you should provide the logger.

        Returns:
            float: The FID score.
        """

        if (logger is None and netG is None) or (
            logger is not None and netG is not None
        ):
            raise ValueError(
                "Either logger or netG should be provided, not both or none."
            )

        if netG is None:
            netG, device = cls.load_generator(logger)

        # ngpu = logger["ngpu"]
        # nlabels = logger["nlabels"]
        # device = logger["device"]
        # nz = logger["nz"]
        # nc = logger["nc"]
        # ngf = logger["ngf"]
        # ndf = logger["ndf"]
        # shared_layers = logger["shared_layers"]
        # n_labels = logger["nlabels"]

        # netG, _ = create_models(
        #     ngpu=ngpu,
        #     nlabels=nlabels,
        #     device=device,
        #     nz=nz,
        #     nc=nc,
        #     ngf=ngf,
        #     ndf=ndf,
        #     shared_layers=shared_layers,
        # )
        # netG.load_state_dict(
        #     torch.load(logger.get_files("g_files")[-1].file_path, weights_only=True)
        # )
        # netG.eval()
        dataset_iter = iter(cls.__initiate_dataset(batch_size))
        fid_model = FrechetInceptionDistance().to(device)

        for _ in trange(num_epochs, desc="calculating FID"):
            fake = (
                netG(
                    netG.make_sample_input(batch_size, device=device),
                    netG.make_sample_labels(batch_size, device=device),
                )
                .detach()
                .repeat_interleave(3, 1)
            )
            # we know the output is of shape (batch_size, 1, h , w) so we need to repeat the channel 3 times to have (batch_size, 3, h, w) shape.
            # why we did that?
            # because the FID model(to be more specific, the inception model) expects the input to be of shape (batch_size, 3, h, w) (3 channels imgs)
            # also, we should map the images between [0, 1] interval
            fid_model.update(
                cls.__uniform_normalize(fake).type(torch.uint8),
                real=False,
            )
            # collect real images
            imgs, _ = next(dataset_iter)
            real_update = (
                cls.__uniform_normalize(imgs.permute(0, 3, 1, 2))
                .repeat_interleave(3, 1)
                .type(torch.uint8)
                .to(device)
            )
            # NOTE: to save some memory, we didnt transfered the whole dataset to the device, we just transferred the current batch.
            assert (
                real_update.shape == (batch_size, 3, 299, 299)
            ), f"{real_update.shape=}, it should be {(batch_size, 3, 299, 299)}, {imgs.shape=}"

            fid_model.update(
                real_update,
                real=True,
            )
        with torch.no_grad():
            fid_score = fid_model.compute()
        return fid_score.item()

    @staticmethod
    def __uniform_normalize(x: torch.Tensor, min_val=0, max_val=255):
        x = x.clone().float()
        xmin = x.min()
        xmax = x.max()
        new_x = min_val + (x - xmin) * (max_val - min_val) / (xmax - xmin)
        assert (
            torch.isclose(new_x.min(), torch.tensor(min_val, dtype=torch.float32))
            and torch.isclose(new_x.max(), torch.tensor(max_val, dtype=torch.float32))
        ), f"new min value should be {min_val} not {new_x.min()=}, new max value should be {max_val} not {new_x.max()=}"
        return torch_interpolate(new_x, size=299, mode="bilinear")

    @classmethod
    def inception_score(
        cls,
        num_epochs: int = 100,
        batch_size: int = 10,
        *,
        device: torch.device | str = None,
        logger: Logger = None,
        netG: Generator = None,
    ) -> tuple[float, float]:
        """
        Calculate the Inception score for the generated images.

        Args:
            num_epochs (int, optional): The number of epochs to generate the images. Defaults to 100.
            batch_size (int, optional): The batch size to use for the FID calculation.
                                        (Therefor, final number of used image will be num_epochs * batch_size)s. Defaults to 10.
            device (torch.device|str, optional): The device to use for the calculation.
                                                  if you are using the logger (netG==None),
                                                  it will be loaded from the logger and this will be ignored. if None, it will use GPU if available. Defaults to None.
            logger (Logger, optional): The logger object to load the generator model from. if None, you should provide the netG.
            netG (Generator, optional): The generator model to use for the calculation. if None, you should provide the logger.

        Returns:
            float: The Inception score.
        """
        if (logger is None and netG is None) or (
            logger is not None and netG is not None
        ):
            raise ValueError(
                "Either logger or netG should be provided, not both or none."
            )

        if netG is None:
            netG, device = cls.load_generator(logger)

        inception_score = InceptionScore().to(device)

        for _ in trange(num_epochs, desc="Calculating the Inception Score"):
            fake = (
                netG(
                    netG.make_sample_input(batch_size, device=device),
                    netG.make_sample_labels(batch_size, device=device),
                )
                .detach()
                .repeat_interleave(3, 1)
            )
            assert fake.ndim == 4 and fake.shape[:2] == (
                batch_size,
                3,
            ), f"{fake.shape=}, it should be ({batch_size}, 3, h, w)"

            new_update = (
                cls.__uniform_normalize(fake, min_val=0, max_val=255)
                .type(torch.uint8)
                .to(device)
            )
            assert (
                new_update.min() >= 0 and new_update.max() <= 255
            ), f"{new_update.min()=} {new_update.max()=}"
            assert new_update.shape == (
                batch_size,
                3,
                299,
                299,
            ), f"{new_update.shape=}, it should be {(batch_size, 3, 299, 299)}"
            assert (
                new_update.dtype == torch.uint8
            ), f"{new_update.dtype=}, it should be torch.uint8"
            inception_score.update(new_update)
            # There is no need to collect real images for the inception score calculation.

        with torch.no_grad():
            return inception_score.compute()

    def show_sample(
        self, run_id: int = -1, count: int = 5, save: bool = False, prefix: str = None
    ):
        """
        Do forward pass on the generator model and display the generated images.

        Args:
        run_id (int, optional): The ID of the model to load and generate images from. if -1, it will load the last model. Defaults to -1.
        count (int, optional): Number of images to generate. so the output will be number_of_subgenerators X count images. Defaults to 5.
        save (bool, optional): If True, it will save the images. Defaults to False.
        prefix (str, optional): The prefix of the saved image files. This will be ignored if save is False. Defaults to None.

        Raises:
        ValueError: If the run_id is not found in the data file.

        Returns:
        The generated images. it will be a matrix of images of shape (number_of_subgenerators, count)
        """
        raise RuntimeError("This function is not implemented yet.")
        print(f"Loading the data file {DATAFILE}.json...")
        with open(os.path.join(self.checkpoint_dir, f"{DATAFILE}.json")) as f:
            all_data = json.load(f)
        if run_id == -1:
            run_id = max(all_data.keys(), key=lambda x: int(x.split("_")[1]))
        if isinstance(run_id, int):
            run_id = f"run_{run_id}"
        run_data = all_data.get(run_id)
        if run_data is None:
            raise ValueError(
                f"The {run_id=} is not found in the data file, available runs are: {all_data.keys()}"
            )
        last_g_name = run_data["g_files"][-1]
        netG, _ = create_models(
            ngpu=run_data["ngpu"],
            nlabels=(n_labels := run_data["nlabels"]),
            device=(device := run_data["device"]),
            nz=(nz := run_data["nz"]),
            nc=run_data["nc"],
            ngf=run_data["ngf"],
            ndf=run_data["ndf"],
            shared_layers=run_data["shared_layers"],
        )
        netG.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, last_g_name)))
        netG.eval()
        assert len(netG.models) == n_labels
        print(f"A generator model with {n_labels} sub-generators is loaded.")
        imgs = []

        for i in range(n_labels):
            img = (
                netG(netG.make_sample_input(count, device=device), torch.tensor(i))
                .detach()
                .cpu()
                .numpy()
            )
            imgs.append(np.concatenate(np.rollaxis(img, 1, 4), axis=1))
        print([img.shape for img in imgs])
        imgs = np.concatenate(imgs, axis=0)
        if save:
            plt.imsave(f"{prefix}_{run_id}_generated.png", imgs)
        else:
            if imgs.shape[-1] == 1:
                plt.imshow(imgs, cmap="gray")
            else:
                plt.imshow(imgs)
            plt.axis("off")
            plt.show()
        return imgs


if __name__ == "__main__":
    # fire.Fire(Evaluate)
    # a = Evaluate().fid(1, num_epochs=10)
    # print(a)
    a = Evaluate()
    print(
        a.inception_score(
            logger=Logger("checkpoints", run_id="y8xer2yz", resume=True), num_epochs=10
        )
    )

    print(
        a.fid(
            logger=Logger("checkpoints", run_id="y8xer2yz", resume=True), num_epochs=10
        )
    )
