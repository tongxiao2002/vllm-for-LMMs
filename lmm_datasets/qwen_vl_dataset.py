import tqdm
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
from transformers import AutoTokenizer, PreTrainedTokenizer
from .base_dataset import MultimodalDataset
from vllm.multimodal.image import ImagePixelData
from .dataset_args import DatasetArgs


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    """Copy from https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/qwen_generation_utils.py#L119
    """

    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


class MultimodalDatasetForQwenVL(MultimodalDataset):
    def __init__(self, dataset_args: DatasetArgs, model_path: str, *args, **kwargs):
        super().__init__(dataset_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.system = "You are a helpful assistant."
        self.image_size = 448

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.trans = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _prepare_vllm_data_and_args(self):
        vllm_data = []
        vllm_args = {
            "llm_args": {
                "disable_image_processor": True,
                "max_model_len": 2048,
                "image_input_type": "pixel_values",
                "image_token_id": 151857,
                "image_input_shape": f"1,3,{self.image_size},{self.image_size}",
                "image_feature_size": 256,
                "dtype": "bfloat16",
            },
            "sampling_args": {
                "stop_tokens": ["<|endoftext|>", "<|im_end|>"],
            },
        }
        for item in tqdm.tqdm(self.instruction_data, ncols=100, desc="Loading data..."):
            question = item['question']
            image = item['image']
            image_transformed = self.transform(image=image)
            prompt = f"Picture1 : <img>fake image</img>\n{question}"
            vllm_data.append({
                "prompt": make_context(tokenizer=self.tokenizer, query=prompt, history=[], system=self.system)[0],
                "multi_modal_data": ImagePixelData(image_transformed)
            })
        # vllm_data = [{}]
        # prompt = "Picture 1: <img>fake image</img>\nBased on the photo, which floor is the Department of Otorhinolaryngology on?"
        # vllm_data[-1] = {
        #     "prompt": make_context(tokenizer=self.tokenizer, query=prompt, history=[], system=self.system)[0],
        #     "multi_modal_data": ImagePixelData(VisionTransformer.transform(Image.open("Hospital_Small.jpg"), image_size=448)),
        # }
        return vllm_args, {"prompts": vllm_data}

    def transform(self, image: Image.Image):
        return self.trans(image).unsqueeze(0)
