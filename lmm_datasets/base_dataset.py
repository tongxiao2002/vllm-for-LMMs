import os
import json
from PIL import Image
from torch.utils.data import Dataset
from .dataset_args import DatasetArgs

problem_solving_prompt = (
    "Please solve the geometry problem based on the following problem text and diagram. "
    "Let's think step by step.\n"
    "Question: {question}\nAnswer: "
)

diagram_description_prompt = (
    "Please describe the diagram of the geometry problem in detail."
)


class MultimodalDataset(Dataset):
    def __init__(self, dataset_args: DatasetArgs, model_path: str):
        super().__init__()
        self.dataset_name = dataset_args.dataset_name
        self.generate_caption = dataset_args.generate_caption
        self.generate_solution = dataset_args.generate_solution
        self.prompt_provide_choices = dataset_args.prompt_provide_choices
        self.model_path = model_path

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

    @property
    def instruction_data(self):
        instruction_data = []
        for item in self.data:
            question = item.pop('question')
            if self.generate_solution:
                prompt = problem_solving_prompt.format(question=question)
            else:
                prompt = diagram_description_prompt
            instruction_data.append({
                "question": prompt,
                **item,
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
