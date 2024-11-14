from pydantic import BaseModel


# @dataclasses.dataclass
class DatasetArgs(BaseModel):
    # name or local path of the dataset
    dataset_name_or_path: str

    # prompts for LLMs
    prompt: str
