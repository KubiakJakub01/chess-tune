"""Supervised fine-tuning operations and utilities for ChessTune."""

import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer

from .utils import log_info


def initialize_new_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    original_vocab_size: int,
):
    """
    Properly initialize embeddings for new tokens to improve training stability.

    This function initializes new token embeddings using the mean and std of existing
    embeddings, which is much more stable than random initialization.
    """
    if len(tokenizer) <= original_vocab_size:
        log_info('No new tokens to initialize')
        return

    num_new_tokens = len(tokenizer) - original_vocab_size
    log_info('Initializing %d new token embeddings', num_new_tokens)

    # Get the embedding layer
    assert hasattr(model, 'get_input_embeddings'), 'Model must have an input embedding layer'
    embedding_layer = model.get_input_embeddings()

    # Calculate mean and std of existing embeddings
    existing_embeddings = embedding_layer.weight.data[:original_vocab_size]
    mean_embedding = existing_embeddings.mean(dim=0)
    std_embedding = existing_embeddings.std(dim=0)

    # Initialize new token embeddings with similar statistics
    with torch.inference_mode():
        for i in range(original_vocab_size, len(tokenizer)):
            # Use normal distribution with mean and std from existing embeddings
            new_embedding = torch.normal(
                mean_embedding, std_embedding * 0.1
            )  # Smaller std for stability
            embedding_layer.weight.data[i] = new_embedding

    log_info('Successfully initialized new token embeddings')


def check_token_embeddings_health(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Check the health of token embeddings, especially new tokens.
    Reports statistics that can help identify training instabilities.
    """
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer.weight.data

    # Basic statistics
    log_info('=== Token Embedding Health Check ===')
    log_info('Total vocabulary size: %d', len(tokenizer))
    log_info('Embedding dimension: %d', embeddings.shape[1])

    # Check for NaN or infinite values
    nan_count = torch.isnan(embeddings).sum().item()
    inf_count = torch.isinf(embeddings).sum().item()

    if nan_count > 0:
        log_info('⚠️  Found %d NaN values in embeddings!', nan_count)
    else:
        log_info('✓ No NaN values in embeddings')

    if inf_count > 0:
        log_info('⚠️  Found %d infinite values in embeddings!', inf_count)
    else:
        log_info('✓ No infinite values in embeddings')

    # Statistics
    mean_norm = embeddings.norm(dim=1).mean().item()
    std_norm = embeddings.norm(dim=1).std().item()
    min_norm = embeddings.norm(dim=1).min().item()
    max_norm = embeddings.norm(dim=1).max().item()

    log_info(
        'Embedding norms - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f',
        mean_norm,
        std_norm,
        min_norm,
        max_norm,
    )

    # Check if any embeddings are too large or too small
    if max_norm > 10.0:
        log_info('⚠️  Some embeddings have very large norms (>10.0)')
    if min_norm < 1e-6:
        log_info('⚠️  Some embeddings have very small norms (<1e-6)')

    log_info('=== End Health Check ===')


def build_optimizer(
    model: PreTrainedModel,
    base_learning_rate: float,
    weight_decay: float,
    embed_lr_multiplier: float = 10.0,
):
    """Return an AdamW optimiser with a separate (higher) LR for token embeddings.

    Args:
        model: The *full* model whose parameters should be optimised.
        base_learning_rate: LR for *most* parameters.
        weight_decay: Weight-decay factor for all parameter groups.
        embed_lr_multiplier: Multiplier applied to *base_learning_rate* for the
            input-embedding matrix.  A value between 5× and 20× usually speeds
            up learning for freshly-initialised tokens without destabilising the
            rest of the network.

    Notes
    -----
    • When LoRA is enabled, only LoRA adapters are trainable by default.  We
      explicitly unfreeze the *token embedding* layer so that new vocabulary
      vectors can receive gradient updates.
    • The function prints how many trainable parameters are present in each
      group, which helps debugging "embeddings not training" issues.
    """
    # Ensure the embedding layer is trainable even when PEFT/LoRA is used.
    embed_layer = model.get_input_embeddings()
    embed_layer.weight.requires_grad = True

    embed_params = list(embed_layer.parameters())
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and 'emb' not in n]

    optimizer = AdamW(
        [
            {
                'params': embed_params,
                'lr': base_learning_rate * embed_lr_multiplier,
            },
            {
                'params': other_params,
                'lr': base_learning_rate,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=weight_decay,
    )

    log_info(
        'Optimizer built: %d embed params @ %.2e LR, %d other params @ %.2e LR',
        sum(p.numel() for p in embed_params),
        base_learning_rate * embed_lr_multiplier,
        sum(p.numel() for p in other_params),
        base_learning_rate,
    )

    return optimizer
