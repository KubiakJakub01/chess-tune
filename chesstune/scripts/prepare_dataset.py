import argparse
import json
import random
from collections import Counter
from collections.abc import Generator
from pathlib import Path

import chess
import chess.pgn

from ..sft_tasks import (
    TASK_REGISTRY,
    SFTTask,
    board_to_fen_conversion,
    fen_to_board_conversion,
    predict_board_after_move,
    predict_custom_token_move,
    predict_san_move,
    square_query_task,
)
from ..tokenizer_ops import (
    BLACK_TURN_TOKEN,
    EMPTY_SQUARE_TOKEN,
    WHITE_TURN_TOKEN,
    BoardRepr,
    MoveRepr,
)
from ..utils import log_info, log_warning


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn_filepath', type=Path, required=True, help='Path to the PGN file.')
    parser.add_argument(
        '--output_filepath', type=Path, required=True, help='Path to the output file.'
    )
    parser.add_argument(
        '--tasks',
        choices=TASK_REGISTRY.keys(),
        default=list(TASK_REGISTRY.keys()),
        nargs='+',
        help='Tasks to include in the dataset.',
    )
    parser.add_argument(
        '--max_records',
        type=int,
        default=None,
        help='Maximum number of records to process.',
    )
    parser.add_argument(
        '--dataset_format',
        type=str,
        default='conversational',
        choices=['conversational', 'instruction'],
        help='Format of the dataset.',
    )
    parser.add_argument(
        '--task_weights',
        type=Path,
        default=None,
        help=(
            'Optional JSON file mapping task names to integer weights.  '
            'A weight >1 duplicates the corresponding training example, '
            'thereby oversampling rare tasks.'
        ),
    )
    return parser.parse_args()


def parse_pgn_file(pgn_filepath: Path) -> Generator[chess.pgn.Game, None, None]:
    """
    Parses a PGN file and yields games.
    """
    with open(pgn_filepath, encoding='utf-8', errors='ignore') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield game


def process_pgn_file(
    pgn_filepath: Path,
    tasks: list[str],
) -> Generator[SFTTask, None, None]:
    """
    Processes a single PGN file and yields training instances.
    Each instance will be a dictionary for SFT.
    """
    processed_games = 0
    for game in parse_pgn_file(pgn_filepath):
        board = game.board()
        sft_records = []

        task_counter = Counter(tasks)

        for node in game.mainline():
            move = node.move
            san_move = board.san(move)

            turn_token = WHITE_TURN_TOKEN if board.turn == chess.WHITE else BLACK_TURN_TOKEN
            board_before_move_tokens = BoardRepr.from_board(board, turn_token)
            move_tokens = MoveRepr.from_move(board.copy(), move)

            # 1. Predict the custom token move
            if task_counter['san_to_custom_move']:
                record = predict_custom_token_move(
                    turn_token=turn_token,
                    san_move=san_move,
                    move_tokens=move_tokens,
                    board_before_move_tokens=board_before_move_tokens,
                )
                sft_records.extend([record] * task_counter['san_to_custom_move'])

            # 2. Board -> SAN move
            if task_counter['board_to_san_move']:
                record = predict_san_move(
                    turn_token=turn_token,
                    san_move=san_move,
                    board_before_move_tokens=board_before_move_tokens,
                )
                sft_records.extend([record] * task_counter['board_to_san_move'])

            # 3. Board + Move -> Resulting board
            if task_counter['board_and_move_to_board']:
                temp_board = board.copy()
                temp_board.push(move)
                board_after_move_tokens = BoardRepr.from_board(temp_board, turn_token)

                record = predict_board_after_move(
                    turn_token=turn_token,
                    san_move=san_move,
                    move_tokens=move_tokens,
                    board_before_move_tokens=board_before_move_tokens,
                    board_after_move_tokens=board_after_move_tokens,
                )
                sft_records.extend([record] * task_counter['board_and_move_to_board'])

            # 4. Board tokens -> FEN and FEN -> Board tokens
            if task_counter['board_to_fen']:
                fen_before = board.fen()

                record = board_to_fen_conversion(
                    board_before_move_tokens=board_before_move_tokens,
                    fen_before=fen_before,
                )
                sft_records.extend([record] * task_counter['board_to_fen'])

            if task_counter['fen_to_board']:
                record = fen_to_board_conversion(
                    fen_before=fen_before,
                    board_before_move_tokens=board_before_move_tokens,
                )
                sft_records.extend([record] * task_counter['fen_to_board'])

            # 5. Square query (random square from board)
            if task_counter['square_query']:
                random_square_index = random.choice(list(chess.SQUARES))
                square_name = chess.square_name(random_square_index)
                piece_at_square = board.piece_at(random_square_index)
                if piece_at_square:
                    answer_token = (
                        'w' if piece_at_square.color == chess.WHITE else 'b'
                    ) + piece_at_square.symbol().upper()
                else:
                    answer_token = EMPTY_SQUARE_TOKEN

                record = square_query_task(
                    board_before_move_tokens=board_before_move_tokens,
                    square_name=square_name,
                    answer_token=answer_token,
                )
                sft_records.extend([record] * task_counter['square_query'])

            try:
                board.push(move)
            except Exception as e:
                log_warning('Error pushing move %s in game from %s: %s', move, pgn_filepath, e)
                break

        processed_games += 1
        yield from sft_records

    log_info('Finished processing %s. Processed %d games.', pgn_filepath, processed_games)


def main(
    pgn_filepath: Path,
    tasks: list[str],
    output_filepath: Path,
    max_records: int | None,
    dataset_format: str,
    task_weights_file: Path | None = None,
):
    expanded_tasks = tasks.copy()
    if task_weights_file and task_weights_file.exists():
        task_weights: dict[str, int] = json.loads(task_weights_file.read_text(encoding='utf-8'))
        log_info('Loaded task weights from %s', task_weights_file)

        expanded_tasks = []
        for task_name in tasks:
            repeats = int(task_weights.get(task_name, 1))
            expanded_tasks.extend([task_name] * max(repeats, 1))

        log_info('Task list expanded for oversampling: %s', expanded_tasks)

    sft_data_count = 0
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    log_info('Starting to process %s and saving to %s...', pgn_filepath, output_filepath)

    with output_filepath.open('w', encoding='utf-8') as f_out:
        for sft_record in process_pgn_file(pgn_filepath, expanded_tasks):
            if dataset_format == 'conversational':
                f_out.write(
                    json.dumps(sft_record.conversational_format(), ensure_ascii=False) + '\n'
                )
            elif dataset_format == 'instruction':
                f_out.write(json.dumps(sft_record.instruction_format(), ensure_ascii=False) + '\n')
            else:
                raise ValueError(f'Invalid dataset format: {dataset_format}')
            sft_data_count += 1
            if sft_data_count % 1000 == 0:
                log_info('Generated %d SFT records...', sft_data_count)
            if max_records is not None and sft_data_count >= max_records:
                log_info('Reached max records %d. Stopping...', max_records)
                break


if __name__ == '__main__':
    args = parse_args()
    main(
        args.pgn_filepath,
        args.tasks,
        args.output_filepath,
        args.max_records,
        args.dataset_format,
        args.task_weights,
    )
