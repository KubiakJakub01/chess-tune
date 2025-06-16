import argparse
from pathlib import Path

from ..tokenizer_ops import ALL_NEW_TOKENS, setup_tokenizer_with_new_tokens
from ..utils import log_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model_name',
        type=str,
        required=True,
        help='Model name to create tokenizer for',
    )
    parser.add_argument(
        '-o',
        '--output_filepath',
        type=Path,
        required=True,
        help='Path to save tokenizer',
    )
    return parser.parse_args()


def main(model_name: str, output_filepath: Path):
    log_info(f'Creating tokenizer for {model_name} and saving to {output_filepath}')
    tokenizer = setup_tokenizer_with_new_tokens(model_name, ALL_NEW_TOKENS)
    log_info(f'Saving tokenizer to {output_filepath}')
    output_filepath.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_filepath)


if __name__ == '__main__':
    args = parse_args()
    main(args.model_name, args.output_filepath)
