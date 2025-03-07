import os


class RunIDNotFoundException(Exception):
    pass


class RunIDAlreadyExistsException(Exception):
    pass


class BaseLogger:
    is_null = None

    def __init__(
        self, run_id: str | int, resume: bool = False
    ):
        # TODO: instead of `resume`, we can use `run_id` to check if the run_id exists or not.
        # TODO: add a `read_only` parameter to the constructor. in the wandb_logger,
        #       if read_only is True, there is no need to call wandb.init and we can use wandb.Api().run(run_id) to check if the run_id exists or not.
        if run_id != -1 and isinstance(run_id, int):
            run_id = f"run_{run_id}"
        if resume:
            if run_id == -1:
                raise ValueError(f"run_id cannot be -1 if resume is True. {run_id=}")
            assert isinstance(
                run_id, str
            ), f"run_id should be a string when resume=True, not {type(run_id)}"
            self.resume_run(run_id)
        else:
            # if run_id is -1, we should create a new run_id
            # otherwise, we should check if the run_id exists
            run_id = self.create_new_run(run_id)

        self.run_id = run_id

    def create_new_run(self, run_id: str | int) -> str:
        """
        This function should create a new run and return the run_id.
        if run_id is -1, it should create a new run and return the run_id.
        if run_id is not -1, it should check if the run_id exists:
            if it exists, it should raise RunIDAlreadyExistsException
            if it doesnt exist, it should create a new run with that run_id

        Raises:
            NotImplementedError: _description_

        Returns:
            str: the run_id
        """
        if run_id == -1:
            new_run = self.create_new_run_with_new_id()

        elif self.if_run_id_exists(run_id):
            raise RunIDAlreadyExistsException(
                f"The run_id {run_id} already exists in the data file."
            )
        else:
            assert isinstance(
                run_id, str
            ), f"run_id should be a string when resume=False and run_id!=-1, not {type(run_id)}"
            new_run = self.create_new_run_with_id(run_id)
            assert new_run == run_id, f"{new_run=} {run_id=}"

        return new_run

    def resume_run(self, run_id: str | int):
        """
        This function should resume the run with the given run_id.
        if the run_id does not exist, it should raise RunIDNotFoundException

        Args:
            run_id (str | int): the run_id of the model.

        Raises:
            NotImplementedError: _description_
        """

        assert isinstance(
            run_id, str
        ), f"run_id should be a string or int, not {type(run_id)}"

        if not self.if_run_id_exists(run_id):
            raise RunIDNotFoundException(f"Run ID {run_id} not found in the data file.")

        self.resume_run_(run_id)

    def if_run_id_exists(self, run_id: str | int):
        """
        This function should check if the run_id exists in the data file.

         Args:
             run_id (str | int): the run_id of the model.

         Raises:
             NotImplementedError: _description_
        """
        raise NotImplementedError

    def resume_run_(self, run_id: str):
        """
        This function should resume the run with the given run_id.

        Args:
            run_id (str): the run_id of the model.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def create_new_run_with_new_id(self) -> str:
        """
        This function should create a new run with a new run_id.

        Raises:
            NotImplementedError: _description_

        Returns:
            str: the run_id
        """
        raise NotImplementedError

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
        raise NotImplementedError
