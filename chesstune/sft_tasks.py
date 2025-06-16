# ruff: noqa: E501
# pylint: disable=line-too-long

from collections.abc import Callable

from pydantic import BaseModel

from .tokenizer_ops import BoardRepr, MoveRepr

TASK_REGISTRY: dict[str, Callable] = {}


def _register(name: str):
    """Decorator to register a task builder function under a given name."""

    def decorator(func):
        TASK_REGISTRY[name] = func
        return func

    return decorator


class SFTTask(BaseModel):
    instruction_text: str
    output_text: str

    def conversational_format(self) -> dict:
        return {
            'messages': [
                {'role': 'user', 'content': self.instruction_text},
                {'role': 'assistant', 'content': self.output_text},
            ]
        }

    def instruction_format(self) -> dict:
        return {
            'prompt': self.instruction_text,
            'completion': self.output_text,
        }


@_register('san_to_custom_move')
def predict_custom_token_move(
    turn_token: str,
    san_move: str,
    move_tokens: MoveRepr,
    board_before_move_tokens: BoardRepr,
) -> SFTTask:
    """
    Predicts the custom token move for a given board and move.
    """
    instruction_text = (
        f'Given the chess board state represented by our custom tokens: '
        f'{board_before_move_tokens.to_string()}'
        f"It is {turn_token}'s turn. "
        f"The next move in Standard Algebraic Notation (SAN) is '{san_move}'. "
        f'What is this move in our custom token format?'
    )
    output_text = move_tokens.to_string()

    return SFTTask(instruction_text=instruction_text, output_text=output_text)


@_register('board_to_san_move')
def predict_san_move(
    turn_token: str,
    san_move: str,
    board_before_move_tokens: BoardRepr,
) -> SFTTask:
    """Prompt that asks the model to output the SAN move given custom board tokens."""
    instruction_text = (
        'Analyze the following chess board state, represented by our custom tokens: '
        f'{board_before_move_tokens.to_string()}'
        f"It is currently {turn_token}'s turn. "
        'What is the next move in Standard Algebraic Notation (SAN)?'
    )
    output_text = san_move

    return SFTTask(instruction_text=instruction_text, output_text=output_text)


@_register('board_and_move_to_board')
def predict_board_after_move(
    turn_token: str,
    san_move: str,
    move_tokens: MoveRepr,
    board_before_move_tokens: BoardRepr,
    board_after_move_tokens: BoardRepr,
) -> SFTTask:
    """Prompt asking model to output board tokens after a given move."""
    instruction_text = (
        "Let's play chess. The current board is represented by our custom tokens: "
        f'{board_before_move_tokens.to_string()}'
        f'It is {turn_token}. '
        f"If the side to move plays '{san_move}' (which is {move_tokens.to_string()} in custom tokens), "
        'what will the state of the board be afterwards in custom-token format?'
    )
    output_text = board_after_move_tokens.to_string()

    return SFTTask(instruction_text=instruction_text, output_text=output_text)


@_register('board_to_fen')
def board_to_fen_conversion(
    board_before_move_tokens: BoardRepr,
    fen_before: str,
) -> SFTTask:
    """Convert custom-token board into standard FEN string."""
    instruction_text = (
        'Convert the following chess position from our custom-token representation to FEN notation: '
        f'{board_before_move_tokens.to_string()}'
    )
    output_text = fen_before

    return SFTTask(instruction_text=instruction_text, output_text=output_text)


@_register('fen_to_board')
def fen_to_board_conversion(
    fen_before: str,
    board_before_move_tokens: BoardRepr,
) -> SFTTask:
    """Convert FEN to custom tokens."""
    instruction_text = (
        'The following FEN describes a chess position. Convert it to our custom-token representation: '
        f'{fen_before}'
    )
    output_text = board_before_move_tokens.to_string()

    return SFTTask(instruction_text=instruction_text, output_text=output_text)


@_register('square_query')
def square_query_task(
    board_before_move_tokens: BoardRepr,
    square_name: str,
    answer_token: str,
) -> SFTTask:
    """Ask which piece (or empty) occupies a particular square."""
    instruction_text = (
        'Given the chess board represented with our custom tokens: '
        f'{board_before_move_tokens.to_string()}'
        f'Which token is located on square {square_name}?'
    )
    output_text = answer_token

    return SFTTask(instruction_text=instruction_text, output_text=output_text)
