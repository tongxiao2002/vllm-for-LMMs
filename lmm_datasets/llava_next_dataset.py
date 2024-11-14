import tqdm
from .base_dataset import MultimodalDataset
from .dataset_args import DatasetArgs


class MultimodalDatasetForLlavaNext(MultimodalDataset):
    def __init__(self, dataset_args: DatasetArgs, *args, **kwargs):
        super().__init__(dataset_args)

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
                "prompt": "[INST]" + "<image>" + "\nUSER: " + question + "\nASSISTANT: " + "[\\INST]",
                "multi_modal_data": {"image": image}
            })
        return vllm_args, {"prompts": vllm_data}
