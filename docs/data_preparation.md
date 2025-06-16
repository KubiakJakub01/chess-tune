# Data Preparation

This document outlines the process of preparing the dataset for training the chess LLM. The goal is to convert chess games from PGN format into a series of instructional prompts for Supervised Fine-Tuning (SFT).

## Overview

The primary script for this process is `chesstune/scripts/prepare_dataset.py`. It reads a PGN file containing one or more chess games and for each move, it generates several training examples based on a set of predefined "SFT tasks".

Each task is designed to teach the model a specific aspect of chess, such as understanding piece movements, converting between board representations, or predicting moves. The output is a JSONL file where each line is a complete training sample in the ShareGPT format.

## SFT Tasks

The `prepare_dataset.py` script can generate data for multiple tasks. The available tasks are defined in `chesstune/sft_tasks.py` and can be selected using the `--tasks` command-line argument.

The following tasks are currently implemented:

| Task Name (`--tasks` option) | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `san_to_custom_move`         | Convert a move in SAN to our custom token format. |
| `board_to_san_move`          | Predict the next move in SAN from a board state.  |
| `board_and_move_to_board`    | Predict the resulting board state after a move.   |
| `board_to_fen`               | Convert a custom token board to FEN notation.     |
| `fen_to_board`               | Convert a FEN string to a custom token board.     |
| `square_query`               | Identify the piece on a specific square.          |

### 1. `san_to_custom_move`

*   **Goal**: To teach the model the mapping from a standard chess move to our custom token representation.
*   **Description**: Given a board state and a move in Standard Algebraic Notation (SAN), the model must produce the equivalent move using the defined custom tokens.

**Example JSON record:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Given the chess board state represented by our custom tokens: <board> w_turn w_R_a1 ... b_R_h8 </board>It is w_turn's turn. The next move in Standard Algebraic Notation (SAN) is 'e4'. What is this move in our custom token format?"
    },
    {
      "role": "assistant",
      "content": "\n```move\nw_pawn_e2_e4\n```\n"
    }
  ]
}
```

### 2. `board_to_san_move`

*   **Goal**: To have the model predict the next best move in SAN.
*   **Description**: This is a classic next-move prediction task. Given a board state in custom tokens, the model should output the next move in SAN.

**Example JSON record:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze the following chess board state, represented by our custom tokens: <board> w_turn w_R_a1 ... b_R_h8 </board>It is currently w_turn's turn. What is the next move in Standard Algebraic Notation (SAN)?"
    },
    {
      "role": "assistant",
      "content": "e4"
    }
  ]
}
```

### 3. `board_and_move_to_board`

*   **Goal**: To teach the model the rules of piece movement and board updates.
*   **Description**: The model is given a board state and a move (in both SAN and custom tokens) and is asked to produce the new board state after the move has been played.

**Example JSON record:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Let's play chess. The current board is represented by our custom tokens: <board> w_turn ... </board>It is w_turn. If the side to move plays 'e4' (which is w_pawn_e2_e4 in custom tokens), what will the state of the board be afterwards in custom-token format?"
    },
    {
      "role": "assistant",
      "content": "<board> b_turn w_R_a1 ... w_P_e4 ... b_R_h8 </board>"
    }
  ]
}
```

*Note: Board token strings in examples are simplified for readability.*

### 4. Board ↔ FEN Conversion (`board_to_fen` & `fen_to_board`)

*   **Goal**: Ensure the model's internal representation is translatable to and from the standard FEN representation.
*   **Description**: These two tasks handle the conversion between our custom token board format and the standard Forsyth-Edwards Notation (FEN).

**Example `board_to_fen`:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Convert the following chess position from our custom-token representation to FEN notation: <board> w_turn ... </board>"
    },
    {
      "role": "assistant",
      "content": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }
  ]
}
```

**Example `fen_to_board`:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "The following FEN describes a chess position. Convert it to our custom-token representation: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    },
    {
      "role": "assistant",
      "content": "<board> w_turn w_R_a1 ... b_R_h8 </board>"
    }
  ]
}
```

### 5. `square_query`

*   **Goal**: Test the model's ability to locate and identify pieces on specific squares.
*   **Description**: The model is given a board and a square name (e.g., "e2") and must return the custom token for the piece occupying that square, or `empty_sq` if unoccupied.

**Example JSON record:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Given the chess board represented with our custom tokens: <board> w_turn ... </board>Which token is located on square e2?"
    },
    {
      "role": "assistant",
      "content": "wP"
    }
  ]
}
```

## Dataset Generation Script

### How to Run

The `chesstune/scripts/prepare_dataset.py` script is used to generate the SFT dataset.

**Usage:**

```bash
python -m chesstune.scripts.prepare_dataset \
    --pgn_filepath /path/to/your/games.pgn \
    --output_filepath /path/to/your/dataset.jsonl \
    --tasks san_to_custom_move board_to_san_move \
    --max_records 10000
```

**Command-line Arguments:**

*   `--pgn_filepath` (required): Path to the input PGN file.
*   `--output_filepath` (required): Path to the output JSONL file.
*   `--tasks` (optional): A space-separated list of task names to generate. If omitted, all implemented tasks are used.
*   `--max_records` (optional): The maximum number of SFT records to generate.
