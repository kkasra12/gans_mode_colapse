# This file should not be used!
raise RuntimeError("You should not use this file!\nDo not regret the past. Look to the future.\n\tSOYEN SHAKU")


from datetime import datetime
import json
import os

from .base_logger import BaseLogger, RunIDNotFoundException
from .file import File


class OfflinLogger(BaseLogger):
    is_null = False

    def __init__(
        self, log_path: os.PathLike | str, run_id: str | int, resume: bool = False
    ):
        self.data_file_path = os.path.join(log_path, "data.json")

        if log_path is None:
            raise ValueError("log_path should be provided.")
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
            print(f"Creating the folder {log_path}...")
            self.write_data({})

        if not os.path.exists(self.data_file_path):
            print(f"Creating the data file {self.data_file_path}...")
            self.write_data({})

        super().__init__(log_path, run_id, resume)

    def if_run_id_exists(self, run_id: str | int):
        """
        This function should check if the run_id exists in the data file.

         Args:
             run_id (str | int): the run_id of the model.

         Raises:
             NotImplementedError: _description_
        """
        return run_id in self.read_data()

    def resume_run_(self, run_id: str | int):
        """
        This function should resume the run with the given run_id.

        Args:
            run_id (str | int): the run_id of the model.

        Raises:
            NotImplementedError: _description_
        """
        pass

    def create_new_run_with_new_id(self) -> str:
        """
        This function should create a new run with a new run_id.

        Raises:
            NotImplementedError: _description_

        Returns:
            str: the run_id
        """
        data = self.read_data()
        new_id = max([int(x.split("_")[1]) for x in data.keys()], default=-1) + 1
        print(f"Creating a new run with run_id: run_{new_id}")
        data[f"run_{new_id}"] = {}
        self.write_data(data)
        return f"run_{new_id}"

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
        data = self.read_data()
        assert (
            run_id not in data
        ), f"Run ID {run_id} already exists in the data file, You should not see this assertion error. Please report it."
        data[run_id] = {}
        self.write_data(data)
        return run_id

    def read_data(self):
        with open(self.data_file_path) as f:
            return json.load(f)

    def write_data(self, data: dict):
        with open(self.data_file_path, "w") as f:
            json.dump(data, f)

    def log(self, key: str, value):
        data = self.read_data()
        if self.run_id not in data:
            raise RunIDNotFoundException(
                f"Run ID {self.run_id} not found in the data file."
            )
        if key not in data[self.run_id]:
            data[self.run_id][key] = []
        data[self.run_id][key].append(
            {"value": value, "timestamp": datetime.now().isoformat()}
        )
        self.write_data(data)

    def log_file(
        self,
        file_path: os.PathLike | str,
        category: str,
        metadata: dict,
        description: str,
    ):
        data = self.read_data()
        if self.run_id not in data:
            raise RunIDNotFoundException(
                f"Run ID {self.run_id} not found in the data file."
            )

        if category not in data[self.run_id]:
            data[self.run_id][category] = []
        data[self.run_id][category].append(
            {
                "file_path": file_path,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                **metadata,
            }
        )
        self.write_data(data)

    def __setitem__(self, key, value):
        data = self.read_data()
        data[self.run_id][key] = value
        self.write_data(data)

    def __getitem__(self, key):
        data = self.read_data()
        return data[self.run_id][key]

    def __contains__(self, key):
        data = self.read_data()
        return key in data[self.run_id]

    # TODO: cache the output of `read_data` and `write_data` functions

    def get_files(self, category: str) -> list[File]:
        return self[category]
