import glob
import os
import pickle

import fire
from matplotlib import pyplot as plt
import numpy as np


class Evaluate:
    def __init__(self, checkpoint_dir: os.PathLike | str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir

    def show_imgs(self, id: int = None):
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
        for img in imgs.reshape(imgs.shape[0], -1, imgs.shape[-1]):
            t.append(img)
            if len(t) == 10:
                img = np.concatenate(t, axis=1)
                plt.imshow(img)
                plt.show()
                t = []


if __name__ == "__main__":
    fire.Fire(Evaluate)
