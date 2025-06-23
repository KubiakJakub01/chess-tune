"""Evaluation utilities and callbacks for ChessTune training."""

from typing import Any

from datasets import Dataset

from .utils import log_info


def create_validation_dataset(
    dataset: Dataset, validation_size: int = 100
) -> tuple[Dataset, Dataset]:
    """
    Create a validation split from the training dataset.

    Args:
        dataset: The original training dataset
        validation_size: Size of validation split

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    val_dataset = dataset.take(validation_size)
    train_dataset = dataset.skip(validation_size)

    log_info(f'Created validation dataset of size {validation_size}')

    return train_dataset, val_dataset


def setup_evaluation_for_trainer(
    trainer_args: dict[str, Any],
    dataset: Dataset,
    use_validation_split: bool = True,
    validation_size: int = 100,
    chess_eval_steps: int = 100,
) -> tuple[dict[str, Any], Dataset | None]:
    """
    Setup evaluation configuration for the SFTTrainer.

    Args:
        trainer_args: Dictionary of trainer configuration arguments
        dataset: The training dataset
        use_validation_split: Whether to create a validation split
        validation_size: Size of validation split
        chess_eval_steps: Steps between chess evaluations

    Returns:
        Updated trainer_args and validation dataset (if created)
    """
    val_dataset = None

    if use_validation_split:
        # Create validation split
        train_dataset, val_dataset = create_validation_dataset(dataset, validation_size)

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
                'do_predict': True,
            }
        )

        log_info('Enabled validation evaluation during training')
    else:
        log_info('Using training without validation split')

    return trainer_args, val_dataset
