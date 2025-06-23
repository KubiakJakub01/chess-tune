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

from ..evaluation import setup_evaluation_for_trainer
from ..sft_ops import check_token_embeddings_health, initialize_new_token_embeddings
from ..tokenizer_ops import ALL_NEW_TOKENS, setup_tokenizer_with_new_tokens
from ..train_config import TrainArgs
from ..utils import log_error, log_info, log_warning


def parse_args() -> TrainArgs:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='ChessTune - Supervised fine-tuning')
    parser.add_argument('--config', type=Path, required=True, help='Path to a JSON config file')
    args = parser.parse_args()

    log_info('Loading configuration from %s', args.config)
    return TrainArgs.from_json(args.config)


def prepare_dataset(dataset_id: str):
    """Loads and processes the dataset into a format accepted by SFTTrainer."""
    if Path(dataset_id).expanduser().exists():
        log_info('Loading local JSONL dataset from %s', dataset_id)
        ds = load_dataset('json', data_files=str(dataset_id), split='train', streaming=True)
    else:
        log_info('Loading dataset %s from the 🤗 Hub', dataset_id)
        ds = load_dataset(dataset_id, split='train', streaming=True)

    return ds


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
    dataset = prepare_dataset(args.dataset)

    # Setup evaluation
    trainer_args_dict = {
        'output_dir': str(args.output_dir),
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.grad_accum_steps,
        'learning_rate': args.learning_rate,
        'max_steps': args.num_train_steps,
        'max_seq_length': args.max_seq_length,
        'warmup_ratio': args.warmup_ratio,
        'logging_steps': args.logging_steps,
        'save_strategy': 'steps',
        'push_to_hub': args.push_to_hub,
        'logging_dir': str(args.tensorboard_dir),
        'report_to': ['tensorboard'] if args.enable_tensorboard else [],
        'max_grad_norm': args.max_grad_norm,
        'weight_decay': args.weight_decay,
        'lr_scheduler_type': args.lr_scheduler_type,
        'dataloader_drop_last': True,
        'save_safetensors': True,
        'eval_strategy': 'no',
        'remove_unused_columns': False,
        'include_tokens_per_second': True,
        'bf16': args.bf16,
        'bf16_full_eval': args.bf16,
        'fp16': args.fp16,
        'fp16_full_eval': args.fp16,
    }

    # Setup evaluation if requested
    if args.use_validation_split:
        trainer_args_dict, _ = setup_evaluation_for_trainer(
            trainer_args_dict,
            dataset,
            use_validation_split=args.use_validation_split,
            validation_size=args.validation_size,
            chess_eval_steps=args.eval_steps,
        )
        dataset = trainer_args_dict.pop('train_dataset')
        eval_dataset = trainer_args_dict.pop('eval_dataset')
    else:
        eval_dataset = None

    trainer_cfg = SFTConfig(**trainer_args_dict)

    # Create trainer with evaluation setup
    trainer_kwargs = {
        'model': model,
        'processing_class': tokenizer,
        'train_dataset': dataset,
        'args': trainer_cfg,
    }

    # Add evaluation dataset and metrics if validation is enabled
    if eval_dataset is not None:
        trainer_kwargs['eval_dataset'] = eval_dataset
        log_info('Enabled validation evaluation')

    trainer = SFTTrainer(**trainer_kwargs)

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
