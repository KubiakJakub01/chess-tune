"""Training-specific configuration models for ChessTune."""

from pathlib import Path

from pydantic import BaseModel, Field


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
    epochs: int = 3
    batch_size: int = 2  # per device
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
    load_in_4bit: bool = False

    # Training stability
    weight_decay: float = Field(0.01, description='Weight decay for regularization')
    lr_scheduler_type: str = Field('cosine', description='Learning rate scheduler type')

    # Misc
    wandb_project: str | None = None
    logging_steps: int = 25
    save_strategy: str = 'epoch'
    fp16: bool = True
    push_to_hub: bool = False

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
