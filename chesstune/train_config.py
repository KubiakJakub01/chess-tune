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

    # LoRA
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Memory / precision
    load_in_4bit: bool = False

    # Misc
    wandb_project: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> 'TrainArgs':
        """Load configuration from a JSON file located at *path*."""
        with open(path, encoding='utf-8') as f:
            return cls.model_validate_json(f.read())

    def to_json(self, path: str | Path) -> None:
        """Dump the current config to *path* in UTF-8 encoded JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding='utf-8')
