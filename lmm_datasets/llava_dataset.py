import tqdm
from .base_dataset import MultimodalDataset
from .dataset_args import DatasetArgs


class MultimodalDatasetForLlava(MultimodalDataset):
    def __init__(self, dataset_args: DatasetArgs, *args, **kwargs):
        super().__init__(dataset_args)
        self.system_prompt = (
            "You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
            "You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of tasks using natural language."
            "Follow the instructions carefully and explain your answers in detail."
        )

    def _prepare_vllm_data_and_args(self):
        vllm_data = []
        vllm_args = {
            "llm_args": {
                "max_model_len": 2048,
                "limit_mm_per_prompt": {"image": 1},
                "dtype": "bfloat16",
            },
            "sampling_args": {},
        }
        for item in tqdm.tqdm(self.instruction_data, ncols=100, desc="Loading data..."):
            question = item['question']
            image = item['image']
            vllm_data.append({
                "prompt": "<image>" + "\nUSER: " + question + "\nASSISTANT:",
                "multi_modal_data": {"image": image}
            })
        return vllm_args, {"prompts": vllm_data}
