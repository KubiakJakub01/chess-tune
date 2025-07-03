from typing import Literal

import chess
import chess.pgn
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utils import log_info, log_warning

# Pieces (White, Black)
PIECE_SYMBOLS = ['P', 'N', 'B', 'R', 'Q', 'K']
NEW_TOKENS_PIECES = [f'w{s} ' for s in PIECE_SYMBOLS] + [f'b{s} ' for s in PIECE_SYMBOLS]

# Squares (a1 to h8)
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']
NEW_TOKENS_SQUARES = [f'{f}{r}' for f in FILES for r in RANKS]

# Special Game Tokens
EMPTY_SQUARE_TOKEN = 'empty_sq '
WHITE_TURN_TOKEN = 'w_turn'
BLACK_TURN_TOKEN = 'b_turn'
SHORT_CASTLING_TOKEN = 'O-O'
LONG_CASTLING_TOKEN = 'O-O-O'
BOARD_TOKEN = '<board>'
BOARD_END_TOKEN = '</board>'
MOVE_TOKEN = '<move>'
MOVE_END_TOKEN = '</move>'

NEW_TOKENS_SPECIAL = [
    EMPTY_SQUARE_TOKEN,
    WHITE_TURN_TOKEN,
    BLACK_TURN_TOKEN,
    SHORT_CASTLING_TOKEN,
    LONG_CASTLING_TOKEN,
    BOARD_TOKEN,
    BOARD_END_TOKEN,
    MOVE_TOKEN,
    MOVE_END_TOKEN,
]
ALL_NEW_TOKENS = sorted(list(set(NEW_TOKENS_PIECES + NEW_TOKENS_SQUARES + NEW_TOKENS_SPECIAL)))

# Helper Mappings
PIECE_TO_TOKEN_MAP = {
    chess.Piece(chess.PAWN, chess.WHITE): 'wP ',
    chess.Piece(chess.KNIGHT, chess.WHITE): 'wN ',
    chess.Piece(chess.BISHOP, chess.WHITE): 'wB ',
    chess.Piece(chess.ROOK, chess.WHITE): 'wR ',
    chess.Piece(chess.QUEEN, chess.WHITE): 'wQ ',
    chess.Piece(chess.KING, chess.WHITE): 'wK ',
    chess.Piece(chess.PAWN, chess.BLACK): 'bP ',
    chess.Piece(chess.KNIGHT, chess.BLACK): 'bN ',
    chess.Piece(chess.BISHOP, chess.BLACK): 'bB ',
    chess.Piece(chess.ROOK, chess.BLACK): 'bR ',
    chess.Piece(chess.QUEEN, chess.BLACK): 'bQ ',
    chess.Piece(chess.KING, chess.BLACK): 'bK ',
}


class BoardRepr(BaseModel):
    board: list[str] = Field(description='List of tokens representing the board')
    turn: str = Field(description='Token representing the turn')

    @classmethod
    def from_board(cls, board: chess.Board, turn_token: str) -> 'BoardRepr':
        return cls(board=board_to_custom_token_sequence(board), turn=turn_token)

    def to_string(self) -> str:
        board_str = f'{BOARD_TOKEN}{self.turn}\n'
        for i, token in enumerate(self.board):
            board_str += token
            if (i + 1) % 8 == 0:
                board_str += '\n'
        board_str += f'{BOARD_END_TOKEN}\n'
        return board_str


class MoveRepr(BaseModel):
    move: list[str] = Field(description='List of tokens representing the move')

    @classmethod
    def from_move(cls, board_before_move: chess.Board, move: chess.Move) -> 'MoveRepr':
        return cls(move=move_to_custom_token_sequence(board_before_move, move))

    def to_string(self) -> str:
        move_str = f'{MOVE_TOKEN}\n'
        for token in self.move:
            move_str += token
        move_str += f'{MOVE_END_TOKEN}\n'
        return move_str


def setup_tokenizer_with_new_tokens(
    model_name: str, new_tokens_list: list[str] = ALL_NEW_TOKENS
) -> PreTrainedTokenizer:
    """Loads a tokenizer and adds new tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    num_added_toks = tokenizer.add_tokens(new_tokens_list, special_tokens=False)
    log_info('Added %d new tokens.', num_added_toks)
    log_info('Vocabulary size after adding tokens: %d', len(tokenizer))
    return tokenizer


def board_to_custom_token_sequence(
    board: chess.Board,
    representation: Literal['compact', 'verbose'] = 'compact',
) -> list[str]:
    """Converts a chess.Board object to a sequence of our custom tokens."""
    assert representation in ['compact', 'verbose'], 'Invalid representation'
    token_sequence: list[str] = []
    if representation == 'compact':
        for square_index in chess.SQUARES:
            piece = board.piece_at(square_index)
            if piece:
                token_sequence.append(PIECE_TO_TOKEN_MAP[piece])
            else:
                token_sequence.append(EMPTY_SQUARE_TOKEN)

    elif representation == 'verbose':
        for square_index in chess.SQUARES:
            square_name = chess.square_name(square_index)
            piece = board.piece_at(square_index)
            if piece:
                token_sequence.append(PIECE_TO_TOKEN_MAP[piece])
                token_sequence.append(square_name)
            else:
                token_sequence.append(EMPTY_SQUARE_TOKEN)
                token_sequence.append(square_name)

    return token_sequence


def move_to_custom_token_sequence(board_before_move: chess.Board, move: chess.Move) -> list[str]:
    """
    Converts a chess.Move object to a sequence of custom tokens.
    Example: "wN g1 f3" or "wP e7 e8 wQ" for promotion
    """
    tokens = []
    piece = board_before_move.piece_at(move.from_square)
    if not piece:
        return ['<error_no_piece>']

    tokens.append(PIECE_TO_TOKEN_MAP[piece])
    tokens.append(chess.square_name(move.from_square))
    tokens.append(chess.square_name(move.to_square))

    # Promotion
    if move.promotion:
        promoted_piece = chess.Piece(move.promotion, board_before_move.turn)
        tokens.append(PIECE_TO_TOKEN_MAP[promoted_piece])
    return tokens


def san_move_to_custom_token_sequence(
    board_before_move: chess.Board, san_move_str: str
) -> list[str]:
    """
    Converts a SAN move string to our custom token sequence.
    This is often more robust for castling.
    """
    if san_move_str == 'O-O':
        return [SHORT_CASTLING_TOKEN]
    if san_move_str == 'O-O-O':
        return [LONG_CASTLING_TOKEN]

    try:
        move = board_before_move.parse_san(san_move_str)
        return move_to_custom_token_sequence(board_before_move, move)
    except ValueError as e:
        log_warning('Warning: Could not parse SAN move "%s": %s', san_move_str, e)
        return [f'<parse_error_{san_move_str.replace(" ", "_")}>']
