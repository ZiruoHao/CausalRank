"""严格的单 episode 过拟合门槛。

该测试不是泛化评估；它验证完整 Encoder -> ParentInteraction -> Decoder 与
三项损失是否至少有能力记住一个真实生成 episode。若本测试失败，不应通过
增加训练数据或搜索学习率掩盖模型/梯度问题。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import CausalRankDataset
from train import CausalRankModel, ModelConfig, compute_losses, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CausalRank 单 episode 过拟合测试")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-assets", type=int, default=64)
    parser.add_argument("--max-times", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.95)
    parser.add_argument("--min-pairwise-accuracy", type=float, default=0.95)
    return parser.parse_args()


def crop_episode(batch: dict[str, torch.Tensor], assets: int, times: int) -> None:
    batch["X"] = batch["X"][:, :assets, :times]
    batch["Y"] = batch["Y"][:, :assets, :times]
    batch["asset_mask"] = batch["asset_mask"][:, :assets, :times]
    batch["feature_mask"] = batch["feature_mask"][:, :assets, :times]
    batch["target_mask"] = batch["target_mask"][:, :assets, :times]
    batch["time_padding_mask"] = ~batch["asset_mask"].any(dim=1)


def evaluate(
    model: CausalRankModel,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        outputs, mask = model(batch)
        items = compute_losses(
            outputs, batch["z"], batch["tau_direct"], mask,
            effect_weight=1.0, rank_weight=1.0, positive_class_weight=0.0,
        )
    positive = mask & batch["z"]
    negative = mask & ~batch["z"]
    predicted = outputs["existence_logits"] >= 0
    recall = (predicted & positive).sum().float() / positive.sum().clamp_min(1)
    specificity = ((~predicted) & negative).sum().float() / negative.sum().clamp_min(1)
    return {
        "loss": float(items["loss"]),
        "causal_loss": float(items["causal_num"] / items["causal_den"]),
        "effect_loss": float(items["effect_num"] / items["effect_den"].clamp_min(1)),
        "ranking_loss": float(items["rank_num"] / items["rank_den"].clamp_min(1)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "pairwise_accuracy": float(items["pair_correct"] / items["pair_count"].clamp_min(1)),
        "score_std": float(outputs["log_scores"][mask].std()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260908)
    device = torch.device(args.device)
    dataset = CausalRankDataset(args.data, "train", max_cached_shards=1)
    episode = dataset[0]
    batch = {name: value.unsqueeze(0) for name, value in episode.items()}
    crop_episode(
        batch,
        min(args.max_assets, dataset.num_assets),
        min(args.max_times, dataset.num_times),
    )
    batch = move_batch(batch, device)

    config = ModelConfig(
        market_state_dim=dataset.market_state_dim,
        embedding_dim=16,
        num_heads=2,
        feedforward_dim=32,
        num_inducing_points=4,
        num_seed_vectors=2,
        cross_row_blocks=1,
        cross_column_blocks=1,
        temporal_layers=1,
        max_time_steps=max(args.max_times, 32),
        parent_hidden_dim=32,
        decoder_hidden_dim=32,
        dropout=0.0,
    )
    model = CausalRankModel(config).to(device)
    model.encoder.configure_cross_section_execution((), chunk_size=0, use_checkpoint=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    initial = evaluate(model, batch)
    print("initial", initial, flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs, mask = model(batch)
        losses = compute_losses(
            outputs, batch["z"], batch["tau_direct"], mask,
            effect_weight=1.0, rank_weight=1.0, positive_class_weight=0.0,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"step={step} metrics={evaluate(model, batch)}", flush=True)

    final = evaluate(model, batch)
    if final["balanced_accuracy"] < args.min_balanced_accuracy:
        raise AssertionError(
            f"balanced accuracy {final['balanced_accuracy']:.4f} "
            f"< {args.min_balanced_accuracy:.4f}"
        )
    if final["pairwise_accuracy"] < args.min_pairwise_accuracy:
        raise AssertionError(
            f"pairwise accuracy {final['pairwise_accuracy']:.4f} "
            f"< {args.min_pairwise_accuracy:.4f}"
        )
    if final["ranking_loss"] >= initial["ranking_loss"] * 0.5:
        raise AssertionError("ranking loss 未至少下降 50%")
    print("PASS: 单 episode 过拟合门槛通过", flush=True)


if __name__ == "__main__":
    main()
