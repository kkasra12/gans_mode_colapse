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

    def show_imgs(self, id: int = None, prefix: str = None):
        """
        Display the generated images from the saved checkpoint.

        Args:
            id (int, optional): The ID of the model to load and generate images from. Defaults to None.
            prefix (str, optional): The prefix of the saved image files. if None, it will show the images instead of saving them. Defaults to None.

        Raises:
            ValueError: If the images are not saved in the checkpoint.

        Returns:
            None
        """
        # TODO: use the id to load the model and generate images
        # but now we are not using the id, so we will load the last epoch model
        last_epoch = max(
            glob.glob(os.path.join(self.checkpoint_dir, "vars_*.pkl")),
            key=lambda x: int(x.split("_")[1]),
        )
        with open(last_epoch, "rb") as f:
            saved_vars = pickle.load(f)

        if "imgs" not in saved_vars:
            raise ValueError(
                f"The images are not saved in the checkpoint, the saved vars are: {saved_vars.keys()}"
            )

        imgs = np.array(saved_vars["imgs"])
        print(imgs.shape)
        t = []
        index = 0
        for img in imgs.reshape(imgs.shape[0], -1, imgs.shape[-1]):
            t.append(img)
            if len(t) == 10:
                img = np.concatenate(t, axis=1)
                if prefix:
                    plt.imsave(f"{prefix}_0.png", img)
                    index += 1
                else:
                    plt.imshow(img)
                plt.show()
                t = []

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


if __name__ == "__main__":
    fire.Fire(Evaluate)
    # a = Evaluate().fid(1, num_epochs=10)
    # print(a)
