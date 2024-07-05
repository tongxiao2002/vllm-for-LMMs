import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from .base_dataset import MultimodalDataset
from vllm.multimodal.image import ImagePixelData


class MultimodalDatasetForCogVLM2(MultimodalDataset):
    def __init__(self, dataset_name: str, model_path: str, *args, **kwargs):
        super().__init__(dataset_name, model_path, *args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.image_size = 1344

        self.trans = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _prepare_vllm_data_and_args(self):
        # image_feature_size comes from EVA2CLIPModel
        image_feature_size = (self.image_size // 14 // 2) ** 2 + 2
        image_token_id = 128002

        vllm_data = []
        vllm_args = {
            "llm_args": {
                "disable_image_processor": True,
                "enforce_eager": True,
                "max_model_len": 3000,
                "max_num_batched_tokens": 3000,
                "image_input_type": "pixel_values",
                "image_token_id": image_token_id,
                "image_input_shape": f"1,3,{self.image_size},{self.image_size}",
                "image_feature_size": image_feature_size,
                "dtype": "bfloat16",
            },
            "sampling_args": {},
        }
        for item in tqdm.tqdm(self.instruction_data, ncols=100, desc="Loading data..."):
            question = item['question']
            image = self.transform(item['image'])
            prompt_token_ids = [self.tokenizer.bos_token_id] \
                + [image_token_id] * image_feature_size \
                + self.tokenizer.encode(question, add_special_tokens=False)

            vllm_data.append({
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": ImagePixelData(image)
            })
        return vllm_args, {"prompts": vllm_data}

    def transform(self, image: Image.Image):
        return self.trans(image).unsqueeze(0)
