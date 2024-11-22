from operator import is_
import os

import wandb

from .file import File
from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    is_null = False
    PROJECT_NAME = "gans_mode_colapse"
    # TODO: apparently, if we change the project name, it wont create a new project. Im not sure!
    ENTITY = "kkasra12"
    run_id: str

    def if_run_id_exists(self, run_id: str | int):
        """
        This function should check if the run_id exists in the data file.

         Args:
             run_id (str | int): the run_id of the model.

         Raises:
             NotImplementedError: _description_
        """
        try:
            wandb.Api().run(f"{self.ENTITY}/{self.PROJECT_NAME}/{run_id}")
            return True
        except (wandb.errors.UsageError, wandb.errors.CommError):
            # print(
            #     f"Avaliable runs: {[r.name for r in wandb.Api().runs("/".join([self.ENTITY, self.PROJECT_NAME]))]}"
            # )
            return False

    def resume_run_(self, run_id: str):
        """
        This function should resume the run with the given run_id.

        Args:
            run_id (str): the run_id of the model.

        Raises:
            NotImplementedError: _description_
        """

        self.run = wandb.init(
            project=self.PROJECT_NAME,
            entity=self.ENTITY,
            id=run_id,
            name=run_id,
            config=(
                wandb.Api().run(f"{self.ENTITY}/{self.PROJECT_NAME}/{run_id}").config
            ),
            resume="must",
        )

    def create_new_run_with_new_id(self) -> str:
        """
        This function should create a new run with a new run_id.

        Raises:
            NotImplementedError: _description_

        Returns:
            str: the run_id
        """
        self.run = wandb.init(self.PROJECT_NAME, self.ENTITY)
        return self.run.id

    def create_new_run_with_id(self, run_id: str) -> str:
        """
        This function should create a new run with the given run_id.

        Args:
            run_id (str): the run_id of the model.

        Raises:
            NotImplementedError: _description_

        Returns:
            str: the run_id
        """
        self.run = wandb.init(project=self.PROJECT_NAME, entity=self.ENTITY, id=run_id)
        return self.run.id

    def log(self, key: str, value):
        self.run.log({key: value})

    def log_file(
        self,
        file_path: os.PathLike | str,
        category: str,
        metadata: dict,
        description: str,
    ):
        # TODO: maybe using `type="file"` is not the best idea!
        artifact = wandb.Artifact(
            category,
            type=category,
            metadata=metadata,
            description=description,
        )
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)

    def __setitem__(self, key, value):
        self.run.config[key] = value

    def __getitem__(self, key):
        return self.run.config[key]

    def get_files(self, category: str) -> list[File]:
        """
        This function should return the list of files in the given category.
        """
        # return (
        #     wandb.Api().artifacts(f"{self.ENTITY}/{self.PROJECT_NAME}/{self
        # )
        return self.run.use_artifact(category)
