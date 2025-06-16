"""Supervised fine-tuning entry-point for the ChessTune project."""

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

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
        ds = load_dataset('json', data_files=str(dataset_id), split='train')
    else:
        log_info('Loading dataset %s from the 🤗 Hub', dataset_id)
        ds = load_dataset(dataset_id, split='train')

    return ds


def build_model_and_tokenizer(args: TrainArgs):
    """Loads the base checkpoint and applies optional LoRA adapters."""

    log_info('Loading tokenizer for %s', args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_eos_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_info('Loading model %s', args.base_model)
    model_kwargs: dict[str, Any] = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
    }
    if args.load_in_4bit:
        model_kwargs['load_in_4bit'] = True

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))

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

    trainer_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        logging_dir=str(args.tensorboard_dir),
        report_to=['tensorboard'] if args.enable_tensorboard else [],
    )

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset, args=trainer_cfg
    )

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
