from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snake_game import ACTION_NAMES, FEATURE_NAMES, SnakeEnv, draw_game


INSTALL_HINT = "缺少依赖。请先运行：python -m pip install torch pygame matplotlib"


@dataclass
class DQNConfig:
    board_size: int = 6
    episodes: int = 500
    gamma: float = 0.9
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 20_000
    warmup_steps: int = 200
    target_update_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    eval_episodes: int = 50
    seed: int = 7


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit(INSTALL_HINT) from exc
    return torch, nn, F


def build_q_network(nn):
    return nn.Sequential(
        nn.Linear(len(FEATURE_NAMES), 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, len(ACTION_NAMES)),
    )


def state_vector(env: SnakeEnv) -> np.ndarray:
    return np.array(env.get_state_bits(), dtype=np.float32)


def select_action(policy_net, state: np.ndarray, epsilon: float, torch, device, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(ACTION_NAMES))
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_model(
    policy_net,
    target_net,
    replay: ReplayBuffer,
    optimizer,
    config: DQNConfig,
    torch,
    F,
    device,
) -> float | None:
    if len(replay) < max(config.batch_size, config.warmup_steps):
        return None

    states, actions, rewards, next_states, dones = replay.sample(config.batch_size)
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    current_q = policy_net(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states_t).max(dim=1).values
        target_q = rewards_t + config.gamma * next_q * (1.0 - dones_t)

    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train_dqn(config: DQNConfig, output_dir: Path) -> dict[str, Any]:
    torch, nn, F = require_torch()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(config.seed)

    env = SnakeEnv(board_size=config.board_size, seed=config.seed)
    policy_net = build_q_network(nn).to(device)
    target_net = build_q_network(nn).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity)

    epsilon = config.epsilon_start
    global_step = 0
    rows: list[dict[str, float]] = []
    recent_losses: list[float] = []

    for episode in range(1, config.episodes + 1):
        env.reset()
        state = state_vector(env)
        done = False
        total_reward = 0.0
        episode_losses: list[float] = []

        while not done:
            action = select_action(policy_net, state, epsilon, torch, device, rng)
            result = env.step_relative(action)
            next_state = state_vector(env)
            replay.push(state, action, result.reward, next_state, result.done)
            loss = optimize_model(policy_net, target_net, replay, optimizer, config, torch, F, device)
            if loss is not None:
                episode_losses.append(loss)
                recent_losses.append(loss)
                if len(recent_losses) > 100:
                    recent_losses.pop(0)

            state = next_state
            done = result.done
            total_reward += result.reward
            global_step += 1
            if global_step % config.target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rows.append(
            {
                "episode": float(episode),
                "score": float(env.score),
                "total_reward": float(total_reward),
                "epsilon": float(epsilon),
                "loss": float(np.mean(episode_losses) if episode_losses else 0.0),
            }
        )
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

    evaluation = evaluate_policy(policy_net, config, torch, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "dqn_model.pt"
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "config": asdict(config),
            "features": FEATURE_NAMES,
            "actions": ACTION_NAMES,
        },
        model_path,
    )
    metrics = save_training_outputs(rows, evaluation, config, output_dir, model_path)
    return metrics


def evaluate_policy(policy_net, config: DQNConfig, torch, device) -> dict[str, float]:
    scores: list[float] = []
    rewards: list[float] = []
    rng = random.Random(config.seed + 1000)
    policy_net.eval()
    with torch.no_grad():
        for _ in range(config.eval_episodes):
            env = SnakeEnv(board_size=config.board_size, seed=rng.randrange(1_000_000_000))
            env.reset()
            done = False
            total_reward = 0.0
            while not done:
                state = torch.tensor(state_vector(env), dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(policy_net(state), dim=1).item())
                result = env.step_relative(action)
                total_reward += result.reward
                done = result.done
            scores.append(float(env.score))
            rewards.append(float(total_reward))
    policy_net.train()
    return {
        "episodes": float(config.eval_episodes),
        "mean_score": float(np.mean(scores) if scores else 0.0),
        "max_score": float(max(scores) if scores else 0.0),
        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
    }


def save_training_outputs(
    rows: list[dict[str, float]],
    evaluation: dict[str, float],
    config: DQNConfig,
    output_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    metrics: dict[str, Any] = {
        "config": asdict(config),
        "model_path": str(model_path),
        "state_features": FEATURE_NAMES,
        "actions": ACTION_NAMES,
        "first_50_mean_score": float(np.mean(scores[: min(50, len(scores))]) if scores else 0.0),
        "last_50_mean_score": float(np.mean(scores[-min(50, len(scores)) :]) if scores else 0.0),
        "best_training_score": int(max(scores) if scores else 0),
        "evaluation": evaluation,
    }

    scores_path = output_dir / "training_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "score", "total_reward", "epsilon", "loss"])
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    curve_path = output_dir / "dqn_training_curve.png"
    save_training_curve(rows, curve_path)
    metrics["outputs"] = {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "scores": str(scores_path),
        "curve": str(curve_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def save_training_curve(rows: list[dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    episodes = [row["episode"] for row in rows]
    scores = [row["score"] for row in rows]
    window = min(50, len(scores))
    moving_avg = [
        float(np.mean(scores[max(0, index - window + 1) : index + 1]))
        for index in range(len(scores))
    ]

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(episodes, scores, color="#f18f01", linewidth=1, alpha=0.45, label="episode score")
    plt.plot(episodes, moving_avg, color="#1ba66a", linewidth=2.5, label=f"{window}-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("DQN Snake Training Curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def load_model(model_path: Path):
    torch, nn, _ = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    config = DQNConfig(**checkpoint.get("config", {}))
    model = build_q_network(nn).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, torch, device


def command_train(args: argparse.Namespace) -> None:
    config = DQNConfig(
        board_size=args.board_size,
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
    metrics = train_dqn(config, Path(args.output_dir))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_eval(args: argparse.Namespace) -> None:
    model, config, torch, device = load_model(Path(args.model_path))
    config.eval_episodes = args.eval_episodes
    result = evaluate_policy(model, config, torch, device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_demo(args: argparse.Namespace) -> None:
    try:
        import pygame  # type: ignore
    except ImportError as exc:
        raise SystemExit(INSTALL_HINT) from exc

    model, config, torch, device = load_model(Path(args.model_path))
    board_size = args.board_size or config.board_size
    env = SnakeEnv(board_size=board_size, seed=args.seed)
    pygame.init()
    header_height = 72
    width = board_size * args.cell_size + 40
    height = board_size * args.cell_size + header_height + 20
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake DQN Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 26, bold=True)
    small_font = pygame.font.SysFont("arial", 16)

    running = True
    pause_frames = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    pause_frames = 0

        if env.done:
            pause_frames += 1
            if pause_frames >= args.fps * 2:
                env.reset()
                pause_frames = 0
        else:
            with torch.no_grad():
                state = torch.tensor(state_vector(env), dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(model(state), dim=1).item())
            env.step_relative(action)

        draw_game(screen, pygame, env, args.cell_size, font, small_font)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"PyTorch MLP-DQN 贪吃蛇训练脚本。依赖安装：{INSTALL_HINT.split('：', 1)[-1]}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练 DQN 并保存模型和训练日志。")
    train.add_argument("--episodes", type=int, default=500)
    train.add_argument("--board-size", type=int, default=6)
    train.add_argument("--gamma", type=float, default=0.9)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--replay-capacity", type=int, default=20_000)
    train.add_argument("--warmup-steps", type=int, default=200)
    train.add_argument("--target-update-interval", type=int, default=200)
    train.add_argument("--epsilon-start", type=float, default=1.0)
    train.add_argument("--epsilon-min", type=float, default=0.05)
    train.add_argument("--epsilon-decay", type=float, default=0.995)
    train.add_argument("--eval-episodes", type=int, default=50)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--output-dir", default="snake_dqn_outputs")
    train.set_defaults(func=command_train)

    eval_parser = subparsers.add_parser("eval", help="加载 DQN 模型并评估贪心策略。")
    eval_parser.add_argument("--model-path", default="snake_dqn_outputs/dqn_model.pt")
    eval_parser.add_argument("--eval-episodes", type=int, default=50)
    eval_parser.set_defaults(func=command_eval)

    demo = subparsers.add_parser("demo", help="加载 DQN 模型并打开窗口自动演示。")
    demo.add_argument("--model-path", default="snake_dqn_outputs/dqn_model.pt")
    demo.add_argument("--board-size", type=int, default=None)
    demo.add_argument("--fps", type=int, default=10)
    demo.add_argument("--cell-size", type=int, default=64)
    demo.add_argument("--seed", type=int, default=2024)
    demo.set_defaults(func=command_demo)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
