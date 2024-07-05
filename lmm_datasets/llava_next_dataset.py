import tqdm
from vllm.multimodal.image import ImagePixelData
from .base_dataset import MultimodalDataset


class MultimodalDatasetForLlavaNext(MultimodalDataset):
    def __init__(self, dataset_name: str, model_path: str, *args, **kwargs):
        super().__init__(dataset_name, model_path, *args, **kwargs)
        self.image_size = 336

    def _prepare_vllm_data_and_args(self):
        image_feature_size = 1176

        vllm_data = []
        vllm_args = {
            "llm_args": {
                "disable_image_processor": False,
                "max_model_len": 2048,
                "image_input_type": "pixel_values",
                "image_token_id": 32000,
                "image_input_shape": f"1,3,{self.image_size},{self.image_size}",
                "image_feature_size": image_feature_size,
                "dtype": "bfloat16",
            },
            "sampling_args": {},
        }

        for item in tqdm.tqdm(self.instruction_data, ncols=100, desc="Loading data..."):
            question = item['question']
            image = item['image']
            vllm_data.append({
                "prompt": "[INST]" + "<image>" * image_feature_size + "\nUSER: " + question + "\nASSISTANT: " + "[\\INST]",
                "multi_modal_data": ImagePixelData(image)
            })
        return vllm_args, {"prompts": vllm_data}
