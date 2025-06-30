"""Supervised fine-tuning entry-point for the ChessTune project."""

import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from trl import SFTConfig, SFTTrainer

from ..callbacks import LogTextSamplesCallback
from ..config import TrainArgs
from ..sft_ops import (
    build_optimizer,
    check_token_embeddings_health,
    initialize_new_token_embeddings,
)
from ..tokenizer_ops import ALL_NEW_TOKENS, setup_tokenizer_with_new_tokens
from ..utils import log_error, log_info, log_warning


def parse_args() -> TrainArgs:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='ChessTune - Supervised fine-tuning')
    parser.add_argument('--config', type=Path, required=True, help='Path to a JSON config file')
    args = parser.parse_args()

    log_info('Loading configuration from %s', args.config)
    return TrainArgs.from_json(args.config, strict=True)


def prepare_dataset(args: TrainArgs) -> tuple[Dataset, Dataset]:
    """Loads and processes the dataset into a format accepted by SFTTrainer."""
    if Path(args.dataset).expanduser().exists():
        log_info('Loading local JSONL dataset from %s', args.dataset)
        ds = load_dataset('json', data_files=str(args.dataset), split='train', streaming=True)
    else:
        log_info('Loading dataset %s from the 🤗 Hub', args.dataset)
        ds = load_dataset(args.dataset, split='train', streaming=True)

    val_dataset = ds.take(args.validation_size)
    train_dataset = ds.skip(args.validation_size)
    train_dataset = train_dataset.shuffle(seed=args.seed, buffer_size=5000)

    return train_dataset, val_dataset


def build_model_and_tokenizer(args: TrainArgs):
    """Loads the base checkpoint and applies optional LoRA adapters."""

    log_info('Loading base tokenizer for %s', args.base_model)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_eos_token=True)
    original_vocab_size = len(base_tokenizer)

    # Setup tokenizer with new tokens
    log_info('Setting up tokenizer with chess tokens')

    tokenizer = setup_tokenizer_with_new_tokens(args.base_model, ALL_NEW_TOKENS)
    tokenizer.padding_side = 'left'

    log_info('Loading model %s', args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **args.model_kwargs)

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    log_info('Model vocab size after resizing: %d', model.config.vocab_size)

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
    set_seed(args.seed)

    if args.wandb_project:
        os.environ['WANDB_PROJECT'] = args.wandb_project

    log_info('Output directory → %s', args.output_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(args)
    train_ds, val_ds = prepare_dataset(args)
    trainer_cfg = SFTConfig(**args.trainer_args)

    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * args.num_train_steps),
        num_training_steps=args.num_train_steps,
    )

    trainer_kwargs = {
        'model': model,
        'processing_class': tokenizer,
        'train_dataset': train_ds,
        'eval_dataset': val_ds,
        'args': trainer_cfg,
        'optimizers': (optimizer, lr_scheduler),
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
