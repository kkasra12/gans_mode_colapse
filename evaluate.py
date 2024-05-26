import glob
import os
import pickle

import fire
from matplotlib import pyplot as plt
import numpy as np


class Evaluate:
    def __init__(self, checkpoint_dir: os.PathLike | str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir

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
            glob.glob(os.path.join(self.checkpoint_dir, f"{prefix}_*.pkl")),
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


if __name__ == "__main__":
    fire.Fire(Evaluate)
