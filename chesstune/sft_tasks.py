# ruff: noqa: E501
# pylint: disable=line-too-long

from collections.abc import Callable

from .tokenizer_ops import BoardRepr

# --- Task Registry Helper --------------------------------------------------
TASK_REGISTRY: dict[str, Callable] = {}


def _register(name: str):
    """Decorator to register a task builder function under a given name."""

    def decorator(func):
        TASK_REGISTRY[name] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# 1. SAN -> Custom-token move
# ---------------------------------------------------------------------------


@_register('san_to_custom_move')
def predict_custom_token_move(
    turn_token: str,
    san_move: str,
    move_tokens: list[str],
    board_before_move_tokens: BoardRepr,
) -> dict:
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
    output_text = f'\n```move\n{" ".join(move_tokens)}\n```\n'

    return {
        'messages': [
            {'role': 'user', 'content': instruction_text},
            {'role': 'assistant', 'content': output_text},
        ]
    }


# ---------------------------------------------------------------------------
# 2. Board tokens -> SAN move (reverse mapping)
# ---------------------------------------------------------------------------


@_register('board_to_san_move')
def predict_san_move(
    turn_token: str,
    san_move: str,
    board_before_move_tokens: BoardRepr,
) -> dict:
    """Prompt that asks the model to output the SAN move given custom board tokens."""
    instruction_text = (
        'Analyze the following chess board state, represented by our custom tokens: '
        f'{board_before_move_tokens.to_string()}'
        f"It is currently {turn_token}'s turn. "
        'What is the next move in Standard Algebraic Notation (SAN)?'
    )
    output_text = san_move

    return {
        'messages': [
            {'role': 'user', 'content': instruction_text},
            {'role': 'assistant', 'content': output_text},
        ]
    }


# ---------------------------------------------------------------------------
# 3. Board + Move -> Resulting board tokens
# ---------------------------------------------------------------------------


@_register('board_and_move_to_board')
def predict_board_after_move(
    turn_token: str,
    san_move: str,
    move_tokens: list[str],
    board_before_move_tokens: BoardRepr,
    board_after_move_tokens: BoardRepr,
) -> dict:
    """Prompt asking model to output board tokens after a given move."""
    instruction_text = (
        "Let's play chess. The current board is represented by our custom tokens: "
        f'{board_before_move_tokens.to_string()}'
        f'It is {turn_token}. '
        f"If the side to move plays '{san_move}' (which is {' '.join(move_tokens)} in custom tokens), "
        'what will the state of the board be afterwards in custom-token format?'
    )
    output_text = board_after_move_tokens.to_string()

    return {
        'messages': [
            {'role': 'user', 'content': instruction_text},
            {'role': 'assistant', 'content': output_text},
        ]
    }


# ---------------------------------------------------------------------------
# 4. Board tokens <-> FEN conversion tasks
# ---------------------------------------------------------------------------


@_register('board_to_fen')
def board_to_fen_conversion(
    board_before_move_tokens: BoardRepr,
    fen_before: str,
) -> dict:
    """Convert custom-token board into standard FEN string."""
    instruction_text = (
        'Convert the following chess position from our custom-token representation to FEN notation: '
        f'{board_before_move_tokens.to_string()}'
    )
    output_text = fen_before

    return {
        'messages': [
            {'role': 'user', 'content': instruction_text},
            {'role': 'assistant', 'content': output_text},
        ]
    }


@_register('fen_to_board')
def fen_to_board_conversion(
    fen_before: str,
    board_before_move_tokens: BoardRepr,
) -> dict:
    """Convert FEN to custom tokens."""
    instruction_text = (
        'The following FEN describes a chess position. Convert it to our custom-token representation: '
        f'{fen_before}'
    )
    output_text = board_before_move_tokens.to_string()

    return {
        'messages': [
            {'role': 'user', 'content': instruction_text},
            {'role': 'assistant', 'content': output_text},
        ]
    }


# ---------------------------------------------------------------------------
# 5. Square-level query task
# ---------------------------------------------------------------------------


@_register('square_query')
def square_query_task(
    board_before_move_tokens: BoardRepr,
    square_name: str,
    answer_token: str,
) -> dict:
    """Ask which piece (or empty) occupies a particular square."""
    instruction_text = (
        'Given the chess board represented with our custom tokens: '
        f'{board_before_move_tokens.to_string()}'
        f'Which token is located on square {square_name}?'
    )
    output_text = answer_token

    return {
        'messages': [
            {'role': 'user', 'content': instruction_text},
            {'role': 'assistant', 'content': output_text},
        ]
    }
