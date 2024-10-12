import tqdm
from transformers import AutoTokenizer
from vllm.multimodal.image import ImagePixelData
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
        self.image_size = 448
        self.patch_size = 14
        self.downsample_ratio = 0.5
        self.img_start_token = '<img>'
        self.img_end_token = '</img>'
        self.img_context_token = '<IMG_CONTEXT>'
        self.template = get_conv_template('Hermes-2')

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.img_context_token)

    def _prepare_vllm_data_and_args(self):
        image_feature_size = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio ** 2))

        vllm_data = []
        vllm_args = {
            "llm_args": {
                "disable_image_processor": False,
                "max_model_len": 8192,
                "image_input_type": "pixel_values",
                "image_token_id": self.image_token_id,
                "image_input_shape": f"1,3,{self.image_size},{self.image_size}",
                "image_feature_size": image_feature_size,
                "dtype": "bfloat16",
            },
            "sampling_args": {
                "stop": [self.template.sep],
            },
        }

        image_tokens = self.img_start_token + self.img_context_token * image_feature_size + self.img_end_token
        for item in tqdm.tqdm(self.instruction_data, ncols=100, desc="Loading data..."):
            question = "<image>\n" + item['question']
            image = item['image']

            template = get_conv_template(self.template.name)
            template.system_message = self.template.system_message
            template.append_message(role=template.roles[0], message=question)
            template.append_message(role=template.roles[1], message=None)
            prompt = template.get_prompt()
            prompt = prompt.replace('<image>', image_tokens, 1)

            model_inputs = self.tokenizer(prompt, return_tensors='pt')
            prompt_token_ids = model_inputs['input_ids'].tolist()[0]

            vllm_data.append({
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": ImagePixelData(image)
            })
        return vllm_args, {"prompts": vllm_data}
