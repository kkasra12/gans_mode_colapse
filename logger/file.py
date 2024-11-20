from wandb import Artifact


class File:
    def __init__(self, input_file: dict | Artifact):
        if isinstance(input_file, dict):
            self.file_path = input_file.pop("file_path")
            self.description = input_file.pop("description")
            self.timestamp = input_file.pop("timestamp")
            self.metadata = input_file
        elif isinstance(input_file, Artifact):
            self.file_path = input_file.get_path()
            self.description = input_file.description
            self.timestamp = input_file.metadata["timestamp"]
            self.metadata = input_file.metadata

    def __eq__(self, other):
        return (
            self.file_path == other.file_path
            and self.description == other.description
            and self.timestamp == other.timestamp
            and self.metadata == other.metadata
        )
