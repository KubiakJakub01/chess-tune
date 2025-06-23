"""Training-specific configuration models for ChessTune."""

from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field
from transformers import BitsAndBytesConfig


class QwenGenerationConfig(BaseModel):
    """Generation configuration for Qwen models."""

    generation_max_new_tokens: int = Field(
        1000, description='Maximum number of new tokens to generate'
    )
    do_sample: bool = Field(True, description='Whether to sample from the model')
    temperature: float = Field(0.6, description='Temperature for sampling')
    top_k: int = Field(20, description='Top-k for sampling')
    top_p: float = Field(0.9, description='Top-p for sampling')


class TrainArgs(BaseModel):
    """Hyper-parameters and paths required for supervised fine-tuning."""

    # Data / IO
    dataset: str = Field(..., description='JSONL file path or 🤗 dataset ID')
    output_dir: Path = Field(Path('models/sft_output'), description='Where to save checkpoints')

    # Base checkpoint
    base_model: str = Field(
        'Qwen/Qwen3-1.7B', description='Local path to base model or Hugging Face model ID'
    )
    tokenizer_path: str = Field(
        'Qwen/Qwen3-1.7B', description='Local path to tokenizer or Hugging Face model ID'
    )

    # Optimisation
    num_train_steps: int = 10000
    batch_size: int = 2
    grad_accum_steps: int = 16
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    max_grad_norm: float = Field(1.0, description='Maximum gradient norm for clipping')

    # LoRA
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Memory / precision
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = 'nf4'
    bnb_4bit_use_double_quant: bool = True

    # Training stability
    weight_decay: float = Field(0.01, description='Weight decay for regularization')
    lr_scheduler_type: str = 'constant'

    # Evaluation
    validation_size: int = Field(100, description='Size of validation split')
    checkpoint_steps: int = Field(100, description='Steps between checkpoints')

    # Misc
    wandb_project: str | None = None
    logging_steps: int = 25
    save_strategy: str = 'epoch'
    fp16: bool = False
    bf16: bool = True
    push_to_hub: bool = False
    attn_implementation: str | None = 'flash_attention_2'
    num_samples_to_log: int = 1
    generation_max_new_tokens: int = 1000

    # TensorBoard / tracking
    enable_tensorboard: bool = True
    tensorboard_dir: Path = Field(
        Path('runs/tensorboard'), description='Directory for TensorBoard logs'
    )

    @classmethod
    def from_json(cls, path: str | Path) -> 'TrainArgs':
        """Load configuration from a JSON file located at *path*."""
        with open(path, encoding='utf-8') as f:
            return cls.model_validate_json(f.read())

    def to_json(self, path: str | Path) -> None:
        """Dump the current config to *path* in UTF-8 encoded JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding='utf-8')

    @property
    def generation_config(self) -> QwenGenerationConfig:
        """Get the generation configuration."""
        return QwenGenerationConfig(generation_max_new_tokens=self.generation_max_new_tokens)

    @property
    def compute_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.bf16 else torch.float16

    @property
    def trainer_args(self) -> dict[str, Any]:
        return {
            'output_dir': str(self.output_dir),
            'per_device_train_batch_size': self.batch_size,
            'per_device_eval_batch_size': self.batch_size,
            'gradient_accumulation_steps': self.grad_accum_steps,
            'learning_rate': self.learning_rate,
            'max_steps': self.num_train_steps,
            'max_seq_length': self.max_seq_length,
            'warmup_ratio': self.warmup_ratio,
            'logging_steps': self.logging_steps,
            'save_strategy': 'steps',
            'push_to_hub': self.push_to_hub,
            'logging_dir': str(self.tensorboard_dir),
            'report_to': ['tensorboard'] if self.enable_tensorboard else [],
            'max_grad_norm': self.max_grad_norm,
            'weight_decay': self.weight_decay,
            'lr_scheduler_type': self.lr_scheduler_type,
            'dataloader_drop_last': True,
            'save_safetensors': True,
            'remove_unused_columns': False,
            'include_tokens_per_second': True,
            'bf16': self.bf16,
            'bf16_full_eval': self.bf16,
            'fp16': self.fp16,
            'fp16_full_eval': self.fp16,
            'eval_strategy': 'steps',
            'eval_steps': self.checkpoint_steps,
            'save_steps': self.checkpoint_steps,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
        }

    @property
    def model_kwargs(self) -> dict[str, Any]:
        bnb_config: BitsAndBytesConfig | None = None
        if self.load_in_4bit or self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )
        return {
            'load_in_8bit': self.load_in_8bit,
            'quantization_config': bnb_config,
            'torch_dtype': self.compute_dtype,
            'device_map': 'auto',
            'attn_implementation': self.attn_implementation,
        }
