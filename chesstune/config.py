from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    """Configuration for the application."""

    model_name: str = 'Qwen/Qwen3-4B-GGUF'
    data_dir: Path = Path('data/pgn')
