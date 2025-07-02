from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import QwenGenerationConfig
from .utils import log_info


def load_model_and_tokenizer(model_path: Path, base_model: str):
    """Load the model and tokenizer for both LoRA and full models."""
    log_info('Loading tokenizer from %s', model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    is_lora = (model_path / 'adapter_config.json').exists()

    if is_lora:
        log_info('Detected LoRA adapter. Loading base model and applying adapter.')
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        log_info('Resizing token embeddings to %d (padded to multiple of 64)', len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, str(model_path))
    else:
        log_info('Detected full model checkpoint. Loading directly from %s', model_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map='auto',
        )

    return model, tokenizer


def run_inference(
    prompt: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: QwenGenerationConfig,
) -> str:
    """Runs inference on the model with a given prompt."""
    messages = [{'role': 'user', 'content': prompt}]
    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
    ).to(model.device)

    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=config.generation_max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )

    outputs = model.generate(tokenized_chat, generation_config=generation_config)
    response_tokens = outputs[0][tokenized_chat.shape[1] :]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    return response_text
