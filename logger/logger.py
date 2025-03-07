import os
import warnings

from sympy import li

from .file import File
from .base_logger import BaseLogger
from .offline_logger import OfflinLogger
from .wandb_logger import WandbLogger


class NullLogger(BaseLogger):
    is_null = True

    def if_run_id_exists(self, run_id: str | int):
        return False

    def resume_run_(self, run_id: str | int):
        pass

    def create_new_run_with_new_id(self) -> str:
        warnings.warn(
            "You are using the NullLogger. "
            "Normally the code flow should not reach here. "
            "Make sure you are using the correct logger. "
        )
        return ""

    def create_new_run_with_id(self, run_id: str | int) -> str:
        return run_id

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return None

    def log(self, key: str, value):
        pass



# TODO: briliant idea: have a list of loggers :D
# PN: Absolut overkilling!
class Logger:
    wandb_logger: NullLogger | WandbLogger

    def __init__(
        self,
        run_id: str | int,
        resume: bool,
    ):
        """

        Args:
            log_folder (os.PathLike | str, optional): the folder to save `data.json` file. if None, it won't save the data file
            run_id (str | int): the run_id of the model. if -1 it will be a new run.
            resume (bool): whether to resume the run or not.
            use_wandb (bool, optional): whther to use wandb or not. Defaults to True.
            lazy (bool, optional): whether to assert the values of the offline and online loggers or not. When True, it will not assert the values. Defaults to True.
        """
        self.wandb_logger = WandbLogger(run_id, resume)
        run_id = self.wandb_logger.run_id
        # if you dont want to use wandb, definitley you will need other alternative
        # self.wandb_logger = NullLogger(log_folder, run_id, resume)

    def log(self, key: str, value):
        self.wandb_logger.log(key, value)

    def log_file(
        self,
        file_path: os.PathLike | str,
        category: str,
        metadata: dict,
        description: str,
    ):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.wandb_logger.log_file(file_path, category, metadata, description)

    def __get_files(self, category: str) -> list[dict]:
        return self.wandb_logger.get_files(category)

    def get_files(self, category: str) -> list[File]:
        return list(map(File, self.__get_files(category)))

    def __setitem__(self, key, value):
        self.wandb_logger[key] = value

    def __getitem__(self, key):
        return self.wandb_logger[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, values: dict):
        for key, value in values.items():
            self[key] = value
        return self
