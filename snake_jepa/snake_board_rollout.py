from __future__ import annotations

from collections import deque

import torch

from snake_jepa.snake_board import EMPTY, FOOD, GRID_SIZE, HEAD, OBSTACLE, SNAKE


MOVES = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


def _snake_cells(board: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, (board.eq(SNAKE) | board.eq(HEAD)).nonzero().tolist()))


def _infer_head(previous: torch.Tensor, current: torch.Tensor, fallback: tuple[int, int] | None) -> tuple[int, int] | None:
    added = list(_snake_cells(current) - _snake_cells(previous))
    if added:
        return added[0]
    cells = list(_snake_cells(current))
    if len(cells) == 1:
        return cells[0]
    return fallback


def initialize_snake_body(boards: list[torch.Tensor]) -> deque[tuple[int, int]]:
    if not boards:
        return deque()

    cells = list(_snake_cells(boards[0]))
    body: deque[tuple[int, int]] = deque(cells[:1])
    for index in range(1, len(boards)):
        current_cells = _snake_cells(boards[index])
        added = list(current_cells - set(body))
        removed = set(body) - current_cells
        if added:
            body.append(added[0])
        if removed:
            body = deque(cell for cell in body if cell not in removed)
        if set(body) != current_cells:
            head = _infer_head(boards[index - 1], boards[index], body[-1] if body else None)
            if head is None:
                body = deque(current_cells)
            else:
                body = deque([cell for cell in current_cells if cell != head] + [head])
    return body


def _current_direction(body: deque[tuple[int, int]]) -> int | None:
    if len(body) < 2:
        return None
    tailward = body[-2]
    head = body[-1]
    dy = (head[0] - tailward[0]) % GRID_SIZE
    dx = (head[1] - tailward[1]) % GRID_SIZE
    if dy == GRID_SIZE - 1 and dx == 0:
        return 0
    if dy == 0 and dx == 1:
        return 1
    if dy == 1 and dx == 0:
        return 2
    if dy == 0 and dx == GRID_SIZE - 1:
        return 3
    return None


def effective_action(body: deque[tuple[int, int]], requested_action: int) -> int:
    current = _current_direction(body)
    if current is not None and int(requested_action) == (current + 2) % 4:
        return current
    return int(requested_action)


def next_head(body: deque[tuple[int, int]], requested_action: int) -> tuple[tuple[int, int] | None, int]:
    if not body:
        return None, int(requested_action)
    action = effective_action(body, int(requested_action))
    dy, dx = MOVES[action]
    head_y, head_x = body[-1]
    return ((head_y + dy) % GRID_SIZE, (head_x + dx) % GRID_SIZE), action


def terminal_transition(
    current_board: torch.Tensor,
    body: deque[tuple[int, int]],
    requested_action: int,
) -> tuple[bool, int, str]:
    candidate, action = next_head(body, requested_action)
    if candidate is None:
        return False, action, ""
    if int(current_board[candidate].item()) == OBSTACLE:
        return True, action, "obstacle"
    grew = int(current_board[candidate].item()) == FOOD
    moving_tail = len(body) > 0 and candidate == body[0] and not grew
    if candidate in body and not moving_tail:
        return True, action, "self"
    return False, action, ""


def legalize_snake_transition(
    current_board: torch.Tensor,
    model_board: torch.Tensor,
    body: deque[tuple[int, int]],
    requested_action: int,
    food_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, deque[tuple[int, int]], int]:
    if not body:
        return model_board.clone(), body, int(requested_action)

    new_head, action = next_head(body, requested_action)
    if new_head is None:
        return model_board.clone(), body, action
    grew = int(current_board[new_head].item()) == FOOD

    next_body = deque(body)
    next_body.append(new_head)
    if not grew:
        next_body.popleft()

    output = torch.full_like(current_board, EMPTY)
    output[current_board.eq(OBSTACLE)] = OBSTACLE

    food: tuple[int, int] | None
    if grew:
        food = next(
            (
                tuple(cell)
                for cell in model_board.eq(FOOD).nonzero().tolist()
                if tuple(cell) not in next_body and int(output[tuple(cell)].item()) != OBSTACLE
            ),
            None,
        )
        if food is None:
            legal_cells = [
                (y, x)
                for y in range(GRID_SIZE)
                for x in range(GRID_SIZE)
                if (y, x) not in next_body and int(output[y, x].item()) == EMPTY
            ]
            if food_scores is not None and legal_cells:
                scores = food_scores.detach().cpu()
                food = max(legal_cells, key=lambda cell: float(scores[cell].item()))
            else:
                food = next(iter(legal_cells), None)
    else:
        current_food = current_board.eq(FOOD).nonzero().tolist()
        food = tuple(current_food[0]) if current_food else None

    if food is not None and food not in next_body and int(output[food].item()) != OBSTACLE:
        output[food] = FOOD
    use_head = bool((current_board.eq(HEAD) | model_board.eq(HEAD)).any().item())
    for cell in (list(next_body)[:-1] if use_head else next_body):
        output[cell] = SNAKE
    if use_head and next_body:
        output[next_body[-1]] = HEAD

    return output, next_body, action
