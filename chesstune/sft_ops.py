"""Supervised fine-tuning utilities for ChessTune."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import log_info


def check_token_embeddings_health(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """
    Check the health of token embeddings, especially new tokens.
    Reports statistics that can help identify training instabilities.
    """
    assert hasattr(model, 'get_input_embeddings'), 'Model must have an input embedding layer'
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer.weight.data

    # Basic statistics
    log_info('=== Token Embedding Health Check ===')
    log_info('Total vocabulary size: %d', len(tokenizer))  # type: ignore
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

    # Check variance within embeddings
    embedding_means = embeddings.mean(dim=0)
    embedding_stds = embeddings.std(dim=0)

    log_info(
        'Per-dimension stats - Mean: %.4f±%.4f, Std: %.4f±%.4f',
        embedding_means.mean().item(),
        embedding_means.std().item(),
        embedding_stds.mean().item(),
        embedding_stds.std().item(),
    )

    log_info('=== End Health Check ===')


def initialize_new_token_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    original_vocab_size: int,
):
    """
    Properly initialize embeddings for new tokens to improve training stability.

    This function initializes new token embeddings using the mean and std of existing
    embeddings, which is much more stable than random initialization.
    """
    if len(tokenizer) <= original_vocab_size:  # type: ignore
        log_info('No new tokens to initialize')
        return

    num_new_tokens = len(tokenizer) - original_vocab_size  # type: ignore
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
        for i in range(original_vocab_size, len(tokenizer)):  # type: ignore
            # Use normal distribution with mean and std from existing embeddings
            new_embedding = torch.normal(
                mean_embedding, std_embedding * 0.1
            )  # Smaller std for stability
            embedding_layer.weight.data[i] = new_embedding

    log_info('Successfully initialized new token embeddings')
