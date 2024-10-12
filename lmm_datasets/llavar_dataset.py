import tqdm
from .base_dataset import MultimodalDataset
from vllm.multimodal.image import ImagePixelData
from .dataset_args import DatasetArgs


class MultimodalDatasetForLlavaR(MultimodalDataset):
    def __init__(self, dataset_args: DatasetArgs, model_path: str, *args, **kwargs):
        super().__init__(dataset_args)
        self.system_prompt = (
            "You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
            "You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of tasks using natural language."
            "Follow the instructions carefully and explain your answers in detail."
        )
        self.image_size = 336

    def _prepare_vllm_data_and_args(self):
        image_feature_size = (self.image_size // 14) ** 2

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
                "prompt": self.system_prompt
                    + " USER: " + question + "\n<im_start>"     # noqa
                    + "<im_patch>" * image_feature_size + "<im_end>" + " ASSISTANT:",   # noqa
                "multi_modal_data": ImagePixelData(image)
            })
        return vllm_args, {"prompts": vllm_data}
