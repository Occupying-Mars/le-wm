from pathlib import Path

import torch
from PIL import Image, ImageDraw


EMPTY = 0
OBSTACLE = 1
SNAKE = 2
FOOD = 3

NUM_BOARD_CLASSES = 4
GRID_SIZE = 20


def extract_board(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        cell_w = image.width // GRID_SIZE
        cell_h = image.height // GRID_SIZE
        board = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.long)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                left = x * cell_w
                top = y * cell_h
                patch = image.crop((left, top, left + cell_w, top + cell_h))
                board[y, x] = classify_cell(patch)
    return board


def classify_cell(patch: Image.Image) -> int:
    pixels = list(patch.getdata())
    if any(r > 180 and g < 80 and b > 160 for r, g, b in pixels):
        return FOOD

    center = pixels[len(pixels) // 2]
    r, g, b = center
    if abs(r - 85) < 25 and abs(g - 85) < 25 and abs(b - 85) < 25:
        return OBSTACLE
    if max(center) < 35:
        return EMPTY
    if g > 120 or r > 120:
        return SNAKE
    return EMPTY


def render_board(board: torch.Tensor, image_size: int) -> Image.Image:
    board = board.detach().cpu()
    cell = image_size // GRID_SIZE
    image = Image.new("RGB", (image_size, image_size), color=(10, 10, 10))
    draw = ImageDraw.Draw(image)

    grid_color = (0, 220, 220)
    for idx in range(GRID_SIZE + 1):
        pos = idx * cell
        draw.line((pos, 0, pos, image_size - 1), fill=grid_color)
        draw.line((0, pos, image_size - 1, pos), fill=grid_color)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            klass = int(board[y, x].item())
            left = x * cell
            top = y * cell
            if klass == OBSTACLE:
                draw.rectangle((left + 1, top + 1, left + cell - 2, top + cell - 2), fill=(85, 85, 85))
            elif klass == SNAKE:
                draw.rectangle((left + 2, top + 2, left + cell - 3, top + cell - 3), fill=(57, 255, 20))
            elif klass == FOOD:
                radius = max(2, cell // 3)
                cx = left + cell // 2
                cy = top + cell // 2
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 16, 240))
                draw.ellipse((cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2), fill=(255, 255, 0))
    return image
