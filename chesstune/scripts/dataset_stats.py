"""Utility script that log_infos token-frequency statistics for a ChessTune JSONL dataset.

Example
-------
$ python -m chesstune.scripts.dataset_stats \
        --dataset data/my_sft.jsonl \
        --tokenizer models/sft_output

The script shows how often each *new* chess token occurs and highlights tokens
that never appear (a red flag for learning efficiency).
"""

import argparse
import json
from collections import Counter
from collections.abc import Generator
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from ..tokenizer_ops import ALL_NEW_TOKENS
from ..utils import log_info


def parse_args():
    parser = argparse.ArgumentParser(description='ChessTune dataset statistics')
    parser.add_argument('--dataset', type=Path, required=True, help='Path to JSONL dataset')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer path or model ID')
    parser.add_argument(
        '--max_records',
        type=int,
        default=None,
        help='Only inspect the first N records (for quick tests)',
    )
    return parser.parse_args()


def yield_text(dataset_path: Path, max_records: int | None) -> Generator[str, None, None]:
    """Yields the *concatenated* text from instruction + answer for each record."""
    with dataset_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            obj = json.loads(line)
            if 'messages' in obj:  # conversational format
                text = '\n'.join(m['content'] for m in obj['messages'])
            else:  # instruction format
                text = obj.get('prompt', '') + '\n' + obj.get('completion', '')
            yield text


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    new_token_ids = {tokenizer.convert_tokens_to_ids(tok): tok for tok in ALL_NEW_TOKENS}

    counter: Counter[int] = Counter()

    log_info('Scanning dataset …')
    for text in tqdm(yield_text(args.dataset, args.max_records)):
        ids = tokenizer(text)['input_ids']
        counter.update(ids)

    log_info('\n=== Token frequencies (new chess tokens only) ===')
    never_seen: list[str] = []
    for tid, tok in sorted(new_token_ids.items(), key=lambda x: x[1]):
        freq = counter[tid]
        if freq == 0:
            never_seen.append(tok)
        log_info(f'{tok:12s}: {freq}')

    if never_seen:
        log_info('\n⚠️  The following tokens were *never* observed – consider oversampling:')
        log_info(', '.join(never_seen))
    else:
        log_info('\n✓ Every new token appears at least once in the dataset.')


if __name__ == '__main__':
    main()
