import os

from sympy import li

from .file import File
from .base_logger import BaseLogger
from .offline_logger import OfflinLogger
from .wandb_logger import WandbLogger


class NullLogger(BaseLogger):
    def if_run_id_exists(self, run_id: str | int):
        return True

    def resume_run_(self, run_id: str | int):
        pass

    def create_new_run_with_new_id(self) -> str:
        return ""

    def create_new_run_with_id(self, run_id: str | int) -> str:
        return ""


class Logger:
    offline_logger: NullLogger | OfflinLogger
    wandb_logger: NullLogger | WandbLogger

    def __init__(
        self,
        log_folder: os.PathLike | str,
        run_id: str | int,
        resume: bool,
        use_wandb: bool = True,
        lazy: bool = True,
    ):
        """

        Args:
            log_folder (os.PathLike | str, optional): the folder to save `data.json` file. if None, it won't save the data file
            run_id (str | int): the run_id of the model. if -1 it will be a new run.
            resume (bool): whether to resume the run or not.
            use_wandb (bool, optional): whther to use wandb or not. Defaults to True.
            lazy (bool, optional): whether to assert the values of the offline and online loggers or not. When True, it will not assert the values. Defaults to True.
        """
        if use_wandb:
            self.wandb_logger = WandbLogger(log_folder, run_id, resume)
            run_id = self.wandb_logger.run_id
            if log_folder is not None:
                self.offline_logger = OfflinLogger(log_folder, run_id, resume)
                self.offline_logger["name"] = self.wandb_logger.run.name
            else:
                self.offline_logger = NullLogger(log_folder, run_id, resume)
        else:
            self.wandb_logger = NullLogger(log_folder, run_id, resume)
            assert (
                log_folder is not None
            ), "you should provide at least one of wandb or offline logger, Am I a joke to you?"
            self.offline_logger = OfflinLogger(log_folder, run_id, resume)

        self.lazy = lazy

    def log(self, key: str, value):
        self.offline_logger.log(key, value)
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
        self.offline_logger.log_file(file_path, category, metadata, description)
        self.wandb_logger.log_file(file_path, category, metadata, description)

    def __get_files(self, category: str) -> list[dict]:
        offline_val = self.offline_logger.get_files(category)
        online_val = None
        if not self.lazy:
            print(f"asserting the values of offline and online loggers... {self.lazy=}")
            online_val = self.wandb_logger.get_files(category)
            if offline_val is not None and online_val is not None:
                assert len(offline_val) == len(
                    online_val
                ), f"{len(offline_val)=} {len(online_val)=}"
                assert all(
                    offline_file == online_file
                    for offline_file, online_file in zip(offline_val, online_val)
                ), (
                    f"some files from {category} category are not the same"
                    f"{offline_val=} {online_val=}"
                )

        if offline_val is not None:
            return offline_val
        elif online_val is not None:
            return online_val
        return self.wandb_logger.get_files(category)

    def get_files(self, category: str) -> list[File]:
        return list(map(File, self.__get_files(category)))

    def __setitem__(self, key, value):
        self.offline_logger[key] = value
        self.wandb_logger[key] = value

    def __getitem__(self, key):
        offline_val = self.offline_logger[key]
        online_val = self.wandb_logger[key]
        if offline_val is not None and online_val is not None:
            assert (
                offline_val == online_val
            ), f"The values are not the same. {offline_val=} {online_val=} {key=}"
        if offline_val is not None:
            return offline_val
        return online_val

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, values: dict):
        for key, value in values.items():
            self[key] = value
        return self
