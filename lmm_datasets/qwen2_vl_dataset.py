import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from .base_dataset import MultimodalDataset, encode_image_base64
from .dataset_args import DatasetArgs


class MultimodalDatasetForQwen2VL(MultimodalDataset):
    def __init__(self, dataset_args: DatasetArgs, model_path: str, *args, **kwargs):
        super().__init__(dataset_args)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _prepare_vllm_data_and_args(self):
        vllm_data = []
        vllm_args = {
            "llm_args": {
                "max_model_len": 8192,
                "limit_mm_per_prompt": {"image": 2},
                "dtype": "bfloat16",
            },
            "sampling_args": {},
        }
        total_msgs = []
        for item in tqdm.tqdm(self.instruction_data, ncols=100, desc="Loading data..."):
            question = item['question']
            images = item['image']
            if isinstance(images, Image.Image):
                images = [images]
            elif isinstance(images, list):
                continue
            else:
                raise ValueError("'image' should only be Image.Image or list of Image.Image.")

            image_msgs = [
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_base64(image)},
                }
                for image in images
            ]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *image_msgs,
                        {"type": "text", "text": question},
                    ],
                }
            ]
            total_msgs.append(messages)
        return vllm_args, {"messages": vllm_data}
