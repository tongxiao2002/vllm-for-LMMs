import tqdm
from transformers import AutoTokenizer
from .conversation import get_conv_template
from ..base_dataset import MultimodalDataset
from ..dataset_args import DatasetArgs


class MultimodalDatasetForInterVL2(MultimodalDataset):
    def __init__(self, dataset_args: DatasetArgs, model_path: str, *args, **kwargs):
        super().__init__(dataset_args)
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.template = get_conv_template('internlm2-chat')

    def _prepare_vllm_data_and_args(self):
        vllm_data = []
        vllm_args = {
            "llm_args": {
                "max_model_len": 8192,
                "limit_mm_per_prompt": {"image": 1},
                "dtype": "bfloat16",
            },
            "sampling_args": {
                "stop": [self.template.sep],
                "stop_token_ids": self.template.stop_token_ids,
            },
        }

        for item in tqdm.tqdm(self.instruction_data, ncols=100, desc="Loading data..."):
            question = "<image>\n" + item['question']
            image = item['image']

            template = get_conv_template(self.template.name)
            template.system_message = self.template.system_message
            template.append_message(role=template.roles[0], message=question)
            template.append_message(role=template.roles[1], message=None)
            prompt = template.get_prompt()

            model_inputs = self.tokenizer(prompt, return_tensors='pt')
            prompt_token_ids = model_inputs['input_ids'].tolist()[0]

            vllm_data.append({
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {"image": image},
            })
        return vllm_args, {"prompts": vllm_data}
