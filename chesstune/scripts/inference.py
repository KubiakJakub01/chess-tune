"""Single-shot inference script for ChessTune SFT models."""

import argparse
from pathlib import Path

from ..config import QwenGenerationConfig
from ..inference_ops import load_model_and_tokenizer, run_inference
from ..sft_tasks import SFTTask
from ..utils import log_error, log_info, log_warning


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ChessTune - SFT Model Inference')
    parser.add_argument(
        '--model_path',
        type=Path,
        required=True,
        help='Path to the fine-tuned model checkpoint (adapter or full model).',
    )
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='Name of the base model (e.g., "meta-llama/Llama-2-7b-hf").',
    )
    parser.add_argument(
        '--input_file',
        type=Path,
        required=True,
        help='Path to the input file containing the tasks.',
    )
    parser.add_argument(
        '--output_file',
        type=Path,
        required=True,
        help='Path to the output file to save the results.',
    )
    parser.add_argument(
        '--max_records',
        type=int,
        default=None,
        help='Maximum number of records to process.',
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1000,
        help='Maximum number of new tokens to generate.',
    )
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p for sampling')
    parser.add_argument(
        '--do_sample', type=bool, default=True, help='Whether to sample from the model'
    )
    return parser.parse_args()


def load_tasks(input_file: Path, max_records: int | None) -> list[SFTTask]:
    """Load the tasks from the input file."""
    with input_file.open(mode='r', encoding='utf-8') as f:
        tasks = [SFTTask.model_validate_json(line) for line in f]
        if max_records is not None:
            return tasks[:max_records]
        return tasks


def save_tasks(tasks: list[SFTTask], output_file: Path):
    """Save the tasks to the output file."""
    with output_file.open(mode='w', encoding='utf-8') as f:
        for task in tasks:
            f.write(task.model_dump_json(indent=2) + '\n')


def main():
    """Main inference script."""
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    tasks = load_tasks(args.input_file, args.max_records)

    generation_config = QwenGenerationConfig(
        generation_max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    for task in tasks:
        prompt = task.instruction_text
        log_info('Generating response for prompt:\n%s', prompt)
        result = run_inference(prompt, model=model, tokenizer=tokenizer, config=generation_config)
        log_info('Result:\n%s', result)
        task.output_text = result
    save_tasks(tasks, args.output_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log_warning('Interrupted by user, exiting …')
    except Exception as exc:
        log_error('Unhandled exception: %s', exc, exc_info=True)
        raise
