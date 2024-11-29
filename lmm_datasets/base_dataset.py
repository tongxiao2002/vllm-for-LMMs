import os
import json
import string
import base64
from io import BytesIO
from PIL import Image
from datasets import (
    Dataset as ArrowDataset,
    load_dataset,
)
from functools import cached_property
from torch.utils.data import Dataset
from .dataset_args import DatasetArgs

kwargs_for_datasets = {
    "mathverse": {
        "path": "data/MathVerse",
        "name": "testmini",
        "split": "testmini",
        "trust_remote_code": True,
        "primary_key": "sample_index",
    }
}


def encode_image_base64(image: Image.Image):
    buffered = BytesIO()
    img_format = image.format
    image.save(buffered, format=img_format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image/{img_format};base64,{img_str}"
    return img_str


class MultimodalDataset(Dataset):
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__()
        self.dataset_name_or_path = dataset_args.dataset_name_or_path
        self.prompt = dataset_args.prompt
        self.dataset_name = os.path.basename(self.dataset_name_or_path).lower()

        assert self.dataset_name in kwargs_for_datasets, "Dataset not found in 'kwargs_for_datasets'."

        dataset_kwargs = kwargs_for_datasets[self.dataset_name]
        self.dataset_primary_key = dataset_kwargs.pop("primary_key")
        self.dataset: ArrowDataset = load_dataset(**dataset_kwargs)
        print(f"Loaded {len(self.dataset)} data samples from '{self.dataset_name_or_path}'.")

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def batch_iter(self, batch_size: int):
        idx = 0
        while idx < len(self.dataset):
            yield self.dataset[idx:idx + batch_size]
            idx += batch_size

    def build_prompt(self, dict_data: dict, **kwargs):
        field_names = [item[1] for item in string.Formatter().parse(self.prompt) if item[1] is not None]
        fields = {**dict_data, **kwargs}
        return self.prompt.format(**{field: fields[field] for field in field_names})

    @cached_property
    def instruction_data(self):
        instruction_data = []
        for item in self.dataset:
            # if "image" in item and "bytes" in item['image']:
            #     buffer = BytesIO(item['image']['bytes'])
            #     item['image'] = Image.open(buffer)
            prompt = self.build_prompt(item)
            instruction_data.append({
                "id": item[self.dataset_primary_key],
                **item,
                "question": prompt,
            })
        return instruction_data

    def collate_fn(self, batch):
        texts = [item['question'] for item in batch]
        images = [item['image'] for item in batch]
        batch_data = self._process_data(texts, images)
        return batch_data

    def _prepare_vllm_data_and_args(self):
        raise NotImplementedError
