import os
import json
import string
import base64
from io import BytesIO
from PIL import Image
from functools import cached_property
from torch.utils.data import Dataset
from .dataset_args import DatasetArgs


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
        self.dataset_name = dataset_args.dataset_name
        self.dataset_path = dataset_args.dataset_path
        self.prompt = dataset_args.prompt
        # self.generate_caption = dataset_args.generate_caption
        # self.generate_solution = dataset_args.generate_solution
        # self.prompt_provide_choices = dataset_args.prompt_provide_choices

        self.data = []
        self.dataset_basedir = os.path.join("data", self.dataset_name, "test")

        for data_id in os.listdir(self.dataset_basedir):
            json_data_path = os.path.join(self.dataset_basedir, data_id, "data.json")
            json_data = json.load(open(json_data_path, "r", encoding="utf-8"))
            image = Image.open(os.path.join(self.dataset_basedir, data_id, "img_diagram.png")).convert('RGB')
            if self.dataset_name.lower() == "geometry3k":
                answer = json_data['precise_value'][ord(json_data['answer'].lower()) - ord('a')]
                question = json_data['problem_text']
            elif self.dataset_name.lower() in ["geoqa", 'geoqa+']:
                answer = json_data['target_number']
                question = json_data['English_problem']
                for number_idx, number in enumerate(json_data['numbers']):
                    question = question.replace(f"N_{number_idx}", str(number))
            else:
                raise ValueError("Only support datasets ['Geometry3K', 'GeoQA', 'GeoQA+'].")
            self.data.append({
                "id": data_id,
                "question": question,
                "image": image,
                "ground-truth": answer,
            })

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def batch_iter(self, batch_size: int):
        idx = 0
        while idx < len(self.data):
            yield self.data[idx:idx + batch_size]
            idx += batch_size

    def build_prompt(self, dict_data: dict, **kwargs):
        field_names = [item[1] for item in string.Formatter().parse(self.prompt) if item[1] is not None]
        fields = {**dict_data, **kwargs}
        return self.prompt.format(**{field: fields[field] for field in field_names})

    @cached_property
    def instruction_data(self):
        instruction_data = []
        for item in self.data:
            prompt = self.build_prompt(item)
            instruction_data.append({
                **item,
                "question": prompt,
            })
        return instruction_data

    def collate_fn(self, batch):
        texts = [item['question'] for item in batch]
        images = [item['image'] for item in batch]
        batch_data = self._process_data(texts, images)
        return batch_data

    def _process_data(self, texts, images):
        raise NotImplementedError

    def _prepare_vllm_data_and_args(self):
        raise NotImplementedError
