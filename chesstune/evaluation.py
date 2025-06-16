"""Evaluation utilities and callbacks for ChessTune training."""

import random
from typing import Any

from datasets import Dataset

from .utils import log_info


def create_validation_dataset(
    training_dataset: Dataset, validation_split: float = 0.1
) -> tuple[Dataset, Dataset]:
    """
    Create a validation split from the training dataset.

    Args:
        training_dataset: The original training dataset
        validation_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Calculate split sizes
    total_size = len(training_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    # Create indices for splitting
    indices = list(range(total_size))

    random.seed(42)
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subset datasets
    train_dataset = training_dataset.select(train_indices)
    val_dataset = training_dataset.select(val_indices)

    log_info(f'Created train/val split: {len(train_dataset)} train, {len(val_dataset)} val samples')

    return train_dataset, val_dataset


def setup_evaluation_for_trainer(
    trainer_args: dict[str, Any],
    dataset: Dataset,
    use_validation_split: bool = True,
    validation_split: float = 0.1,
    chess_eval_steps: int = 100,
) -> tuple[dict[str, Any], Dataset | None]:
    """
    Setup evaluation configuration for the SFTTrainer.

    Args:
        trainer_args: Dictionary of trainer configuration arguments
        dataset: The training dataset
        use_validation_split: Whether to create a validation split
        validation_split: Fraction of data for validation
        chess_eval_steps: Steps between chess evaluations

    Returns:
        Updated trainer_args and validation dataset (if created)
    """
    val_dataset = None

    if use_validation_split:
        # Create validation split
        train_dataset, val_dataset = create_validation_dataset(dataset, validation_split)

        # Update trainer args for evaluation
        trainer_args.update(
            {
                'train_dataset': train_dataset,
                'eval_dataset': val_dataset,
                'eval_strategy': 'steps',
                'eval_steps': chess_eval_steps,
                'save_strategy': 'steps',
                'save_steps': chess_eval_steps,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
            }
        )

        log_info('Enabled validation evaluation during training')
    else:
        log_info('Using training without validation split')

    return trainer_args, val_dataset
