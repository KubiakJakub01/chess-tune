"""Supervised fine-tuning entry-point for the ChessTune project."""

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from ..callbacks import LogTextSamplesCallback
from ..config import TrainArgs
from ..sft_ops import check_token_embeddings_health, initialize_new_token_embeddings
from ..tokenizer_ops import ALL_NEW_TOKENS, setup_tokenizer_with_new_tokens
from ..utils import log_error, log_info, log_warning


def parse_args() -> TrainArgs:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='ChessTune - Supervised fine-tuning')
    parser.add_argument('--config', type=Path, required=True, help='Path to a JSON config file')
    args = parser.parse_args()

    log_info('Loading configuration from %s', args.config)
    return TrainArgs.from_json(args.config)


def prepare_dataset(dataset_id: str, args: TrainArgs):
    """Loads and processes the dataset into a format accepted by SFTTrainer."""
    if Path(dataset_id).expanduser().exists():
        log_info('Loading local JSONL dataset from %s', dataset_id)
        ds = load_dataset('json', data_files=str(dataset_id), split='train', streaming=True)
    else:
        log_info('Loading dataset %s from the 🤗 Hub', dataset_id)
        ds = load_dataset(dataset_id, split='train', streaming=True)

    val_dataset = ds.take(args.validation_size)
    train_dataset = ds.skip(args.validation_size)

    return train_dataset, val_dataset


def build_model_and_tokenizer(args: TrainArgs):
    """Loads the base checkpoint and applies optional LoRA adapters."""

    log_info('Loading base tokenizer for %s', args.base_model)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_eos_token=True)
    original_vocab_size = len(base_tokenizer)

    # Setup tokenizer with new tokens
    log_info('Setting up tokenizer with chess tokens')

    tokenizer = setup_tokenizer_with_new_tokens(args.base_model, ALL_NEW_TOKENS)

    log_info('Loading model %s', args.base_model)
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model_kwargs: dict[str, Any] = {
        'load_in_8bit': args.load_in_8bit,
        'quantization_config': bnb_config,
        'torch_dtype': compute_dtype,
        'device_map': 'auto',
        'attn_implementation': args.attn_implementation,
    }

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Properly initialize new token embeddings
    initialize_new_token_embeddings(model, tokenizer, original_vocab_size)

    # Check embedding health after initialization
    check_token_embeddings_health(model, tokenizer)

    if args.use_lora:
        log_info(
            'Attaching LoRA adapters (r=%d, alpha=%d, dropout=%.2f)',
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
            ],
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    """Main CLI entry-point."""

    args = parse_args()
    if args.wandb_project:
        os.environ['WANDB_PROJECT'] = args.wandb_project

    log_info('Output directory → %s', args.output_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(args)
    train_ds, val_ds = prepare_dataset(args.dataset, args)
    trainer_cfg = SFTConfig(**args.trainer_args)

    trainer_kwargs = {
        'model': model,
        'processing_class': tokenizer,
        'train_dataset': train_ds,
        'eval_dataset': val_ds,
        'args': trainer_cfg,
    }

    trainer = SFTTrainer(**trainer_kwargs)
    text_generation_callback = LogTextSamplesCallback(
        eval_dataset=val_ds,
        args=args,
    )
    trainer.add_callback(text_generation_callback)

    log_info('Starting training …')
    trainer.train()

    log_info('Saving final checkpoint …')
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    log_info('Training completed! Model saved to %s', args.output_dir.resolve())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log_warning('Interrupted by user, exiting …')
    except Exception as exc:  # pylint: disable=broad-except
        log_error('Unhandled exception: %s', exc)
        raise
