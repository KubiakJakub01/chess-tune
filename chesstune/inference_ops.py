from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from .config import TrainArgs


def run_inference(
    prompt: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    args: TrainArgs,
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
        max_new_tokens=args.generation_config.generation_max_new_tokens,
        do_sample=args.generation_config.do_sample,
        temperature=args.generation_config.temperature,
        top_k=args.generation_config.top_k,
        top_p=args.generation_config.top_p,
    )

    outputs = model.generate(tokenized_chat, generation_config=generation_config)
    response_tokens = outputs[0][tokenized_chat.shape[1] :]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    return response_text
