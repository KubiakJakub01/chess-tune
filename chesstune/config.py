from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    """Configuration for the application."""

    model_name: str = 'Qwen/Qwen3-1.7B'
    data_dir: Path = Path('data/pgn')
