from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Literal


UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
DIRECTION_NAMES = ["up", "right", "down", "left"]
ACTION_NAMES = ["straight", "left", "right"]
FEATURE_NAMES = [
    "danger_straight",
    "danger_left",
    "danger_right",
    "dir_up",
    "dir_right",
    "dir_down",
    "dir_left",
    "food_up",
    "food_right",
    "food_down",
    "food_left",
]


EventName = Literal["move", "food", "collision", "timeout", "win"]


@dataclass
class StepResult:
    state_id: int
    reward: float
    done: bool
    event: EventName
    score: int


class SnakeEnv:
    """可被人工游戏和强化学习脚本共用的贪吃蛇环境。"""

    def __init__(
        self,
        board_size: int = 6,
        seed: int | None = None,
        step_penalty: float = -0.1,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        timeout_penalty: float = -5.0,
    ) -> None:
        if board_size < 5:
            raise ValueError("board_size 至少为 5，否则初始蛇身无法稳定放置。")
        self.board_size = board_size
        self.rng = random.Random(seed)
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.timeout_penalty = timeout_penalty
        self.snake: list[tuple[int, int]] = []
        self.direction = RIGHT
        self.food = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.reset()

    @property
    def max_steps_without_food(self) -> int:
        return self.board_size * self.board_size * 2

    def reset(self) -> int:
        center = self.board_size // 2
        self.direction = RIGHT
        self.snake = [(center - 1, center), (center - 2, center), (center - 3, center)]
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.food = self._spawn_food()
        return self.get_state_id()

    def _spawn_food(self) -> tuple[int, int]:
        free_cells = [
            (x, y)
            for y in range(self.board_size)
            for x in range(self.board_size)
            if (x, y) not in self.snake
        ]
        if not free_cells:
            return (-1, -1)
        return self.rng.choice(free_cells)

    def _next_head(self, direction: tuple[int, int]) -> tuple[int, int]:
        head_x, head_y = self.snake[0]
        dir_x, dir_y = direction
        return head_x + dir_x, head_y + dir_y

    def _turn_relative(self, action_id: int) -> tuple[int, int]:
        direction_index = DIRECTIONS.index(self.direction)
        if action_id == 0:
            return self.direction
        if action_id == 1:
            return DIRECTIONS[(direction_index - 1) % len(DIRECTIONS)]
        if action_id == 2:
            return DIRECTIONS[(direction_index + 1) % len(DIRECTIONS)]
        raise ValueError("action_id 必须是 0、1、2，分别表示直行、左转、右转。")

    def _will_collide(self, point: tuple[int, int]) -> bool:
        x, y = point
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return True
        # 普通移动时尾巴会离开，因此允许蛇头进入当前尾巴所在格。
        body_without_tail = set(self.snake[:-1])
        return point in body_without_tail

    def _is_reverse(self, direction: tuple[int, int]) -> bool:
        return direction[0] == -self.direction[0] and direction[1] == -self.direction[1]

    def get_state_bits(self) -> list[int]:
        straight = self.direction
        left = DIRECTIONS[(DIRECTIONS.index(self.direction) - 1) % len(DIRECTIONS)]
        right = DIRECTIONS[(DIRECTIONS.index(self.direction) + 1) % len(DIRECTIONS)]
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return [
            int(self._will_collide(self._next_head(straight))),
            int(self._will_collide(self._next_head(left))),
            int(self._will_collide(self._next_head(right))),
            int(self.direction == UP),
            int(self.direction == RIGHT),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(food_y < head_y),
            int(food_x > head_x),
            int(food_y > head_y),
            int(food_x < head_x),
        ]

    def get_state_id(self) -> int:
        value = 0
        for bit in self.get_state_bits():
            value = (value << 1) | bit
        return value

    def step_relative(self, action_id: int) -> StepResult:
        if self.done:
            return StepResult(self.get_state_id(), 0.0, True, "collision", self.score)
        return self._step(self._turn_relative(action_id))

    def step_direction(self, requested_direction: tuple[int, int]) -> StepResult:
        if requested_direction not in DIRECTIONS:
            raise ValueError("requested_direction 必须来自 DIRECTIONS。")
        if self._is_reverse(requested_direction):
            requested_direction = self.direction
        return self._step(requested_direction)

    def _step(self, new_direction: tuple[int, int]) -> StepResult:
        self.steps += 1
        self.steps_since_food += 1
        self.direction = new_direction
        new_head = self._next_head(self.direction)

        if self._will_collide(new_head):
            self.done = True
            return StepResult(self.get_state_id(), self.death_penalty, True, "collision", self.score)

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            if len(self.snake) == self.board_size * self.board_size:
                self.done = True
                return StepResult(self.get_state_id(), self.food_reward, True, "win", self.score)
            self.food = self._spawn_food()
            return StepResult(self.get_state_id(), self.food_reward, False, "food", self.score)

        self.snake.pop()
        if self.steps_since_food >= self.max_steps_without_food:
            self.done = True
            return StepResult(self.get_state_id(), self.timeout_penalty, True, "timeout", self.score)

        return StepResult(self.get_state_id(), self.step_penalty, False, "move", self.score)


def _load_pygame():
    try:
        import pygame  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "未安装 pygame，无法打开窗口。请先运行：python -m pip install pygame"
        ) from exc
    return pygame


def draw_game(screen, pygame, env: SnakeEnv, cell_size: int, font, small_font) -> None:
    width = env.board_size * cell_size
    header_height = 72
    screen.fill((246, 247, 241))

    title = font.render(f"Snake  Score: {env.score}", True, (18, 48, 59))
    help_text = small_font.render("Arrow/WASD move   R restart   ESC quit", True, (72, 92, 101))
    screen.blit(title, (20, 14))
    screen.blit(help_text, (20, 44))

    board_rect = pygame.Rect(20, header_height, width, width)
    pygame.draw.rect(screen, (255, 255, 255), board_rect, border_radius=8)

    for y in range(env.board_size):
        for x in range(env.board_size):
            color = (235, 240, 242) if (x + y) % 2 == 0 else (226, 233, 236)
            cell = pygame.Rect(20 + x * cell_size, header_height + y * cell_size, cell_size - 2, cell_size - 2)
            pygame.draw.rect(screen, color, cell, border_radius=6)

    food_x, food_y = env.food
    if food_x >= 0:
        food_center = (
            20 + food_x * cell_size + cell_size // 2,
            header_height + food_y * cell_size + cell_size // 2,
        )
        pygame.draw.circle(screen, (241, 143, 1), food_center, max(6, cell_size // 3))

    for index, (x, y) in enumerate(reversed(env.snake)):
        color = (27, 107, 79) if index == len(env.snake) - 1 else (45, 167, 114)
        rect = pygame.Rect(24 + x * cell_size, header_height + y * cell_size + 4, cell_size - 10, cell_size - 10)
        pygame.draw.rect(screen, color, rect, border_radius=7)

    if env.done:
        overlay = pygame.Surface((width, width), pygame.SRCALPHA)
        overlay.fill((18, 48, 59, 165))
        screen.blit(overlay, board_rect)
        message = font.render("Game Over - Press R", True, (255, 255, 255))
        screen.blit(message, message.get_rect(center=board_rect.center))


def run_manual_game(board_size: int, fps: int, cell_size: int, seed: int | None) -> None:
    pygame = _load_pygame()
    pygame.init()
    env = SnakeEnv(board_size=board_size, seed=seed)
    header_height = 72
    width = board_size * cell_size + 40
    height = board_size * cell_size + header_height + 20
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake RL Lab")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 26, bold=True)
    small_font = pygame.font.SysFont("arial", 16)

    key_to_direction = {
        pygame.K_UP: UP,
        pygame.K_w: UP,
        pygame.K_RIGHT: RIGHT,
        pygame.K_d: RIGHT,
        pygame.K_DOWN: DOWN,
        pygame.K_s: DOWN,
        pygame.K_LEFT: LEFT,
        pygame.K_a: LEFT,
    }
    pending_direction = env.direction
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    pending_direction = env.direction
                elif event.key in key_to_direction:
                    pending_direction = key_to_direction[event.key]

        if not env.done:
            env.step_direction(pending_direction)

        draw_game(screen, pygame, env, cell_size, font, small_font)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手动游玩的 pygame 贪吃蛇。")
    parser.add_argument("--board-size", type=int, default=6, help="棋盘大小，默认 6。")
    parser.add_argument("--fps", type=int, default=8, help="游戏刷新速度，默认 8。")
    parser.add_argument("--cell-size", type=int, default=64, help="每个格子的像素大小，默认 64。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认不固定。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_manual_game(args.board_size, args.fps, args.cell_size, args.seed)


if __name__ == "__main__":
    main()
