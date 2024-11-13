from pydantic import BaseModel


# @dataclasses.dataclass
class DatasetArgs(BaseModel):
    # name of the dataset
    dataset_name: str

    # storage path of the dataset
    dataset_path: str

    # prompts for LLMs
    prompt: str
