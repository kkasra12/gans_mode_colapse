import glob
import json
from operator import is_
import os
import pickle

import fire
from matplotlib import pyplot as plt
import numpy as np
import torch

from torcheval.metrics import FrechetInceptionDistance
from tqdm import trange
from data import MnistDataset
from model import create_models

DATAFILE = "data"


class Evaluate:
    def __init__(self, checkpoint_dir: os.PathLike | str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        print("init")

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
        plt.plot(saved_vars["g_losses"][:50], label="Generator Loss")
        plt.plot(saved_vars["d_losses"][:50], label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if prefix:
            plt.savefig(f"{prefix}_{run_id}_loss.png")
        else:
            plt.show()

    def fid(self, run_id: int, num_epochs: int = 100, batch_size: int = 10):
        """
        Calculate the FID score for the generated images.

        Args:
            run_id (int): The ID of the model to load and calculate the FID score.
            num_epochs (int, optional): The number of epochs to generate the images. Defaults to 100.
            batch_size (int, optional): The batch size to use for the FID calculation. (Therefor, final number of used image will be num_epochs * batch_size)s. Defaults to 10.


        Returns:
            None
        """
        print(f"Loading the data file {DATAFILE}.json...")
        with open(os.path.join(self.checkpoint_dir, f"{DATAFILE}.json")) as f:
            run_data = json.load(f).get(f"run_{run_id}")
            if run_data is None:
                raise ValueError("The run_id is not found in the data file.")
            if "fid" in run_data:
                print(f"The FID score for run_{run_id} is: {run_data['fid']}")
                return run_data["fid"]
        print(f"Calculating the FID score for run_{run_id}...")
        # now we need to calculate the FID score
        # we will use the last epoch model
        last_epoch_g = max(run_data["g_files"], key=lambda x: int(x.split("_")[1]))
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
        netG.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, last_epoch_g))
        )
        netG.eval()
        dataset_iter = iter(MnistDataset(DATAFILE, batch_size=batch_size))
        fid_model = FrechetInceptionDistance()

        for _ in trange(num_epochs):
            fake = (
                netG(
                    torch.randn(batch_size, nz, 1, 1, device=device),
                    torch.randint(0, n_labels, (batch_size,)),
                )
                .detach()
                .repeat_interleave(3, 1)
            )
            # we know the output is of shape (batch_size, 1, h , w) so we need to repeat the channel 3 times to have (batch_size, 3, h, w) shape.
            # why we did that?
            # because the FID model(to be more specific, the inception model) expects the input to be of shape (batch_size, 3, h, w) (3 channels imgs)
            # also, we should map the images between [0, 1] interval
            fid_model.update(
                self.__uniform_normalize(fake),
                is_real=False,
            )
            # collect real images
            imgs, _ = next(dataset_iter)
            fid_model.update(
                self.__uniform_normalize(imgs)
                .permute(0, 3, 1, 2)
                .repeat_interleave(3, 1),
                is_real=True,
            )
        with torch.no_grad():
            fid_score = fid_model.compute()
        return fid_score.item()

    def __uniform_normalize(self, x: torch.Tensor):
        return (x - (m := x.min())) / (x.max() - m)

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
        print(f"Loading the data file {DATAFILE}.json...")
        with open(os.path.join(self.checkpoint_dir, f"{DATAFILE}.json")) as f:
            all_data = json.load(f)
        if run_id == -1:
            run_id = max(all_data.keys(), key=lambda x: int(x.split("_")[1]))
        run_data = all_data.get(run_id)
        if run_data is None:
            raise ValueError(f"The {run_id=} is not found in the data file.")
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
    fire.Fire(Evaluate)
    # a = Evaluate().fid(1, num_epochs=10)
    # print(a)
