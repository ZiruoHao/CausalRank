"""CausalRank  训练入口。

链路固定为 Dataset -> PanelEncoder -> ParentInteraction -> RankingDecoder。
支持全合成预训练、半合成微调、可选 30% 全合成 replay、掩码损失和断点续训。
"""

from __future__ import annotations

import json
import random
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

# 兼容项目根目录直接运行和从父目录以 ``python -m CausalRank.train`` 运行。
if __package__:
    from .args import parse_args
    from .modules import (
        CausalRankDataset,
        CausalRankPanelEncoder,
        CausalRankingDecoder,
        TargetAwareParentInteraction,
        derive_factor_observation_mask,
    )
else:
    from args import parse_args
    from modules import (
        CausalRankDataset,
        CausalRankPanelEncoder,
        CausalRankingDecoder,
        TargetAwareParentInteraction,
        derive_factor_observation_mask,
    )


@dataclass(frozen=True)
class ModelConfig:
    market_state_dim: int = 0
    embedding_dim: int = 128
    num_heads: int = 8
    feedforward_dim: int = 256
    num_inducing_points: int = 32
    num_seed_vectors: int = 4
    cross_row_blocks: int = 4
    cross_column_blocks: int = 4
    temporal_layers: int = 2
    max_time_steps: int = 256
    parent_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    dropout: float = 0.1


class CausalRankModel(nn.Module):
    """组合三个可训练模块，并在 episode 级传播因子可观测掩码。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CausalRankPanelEncoder(
            market_state_dim=config.market_state_dim,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            num_inducing_points=config.num_inducing_points,
            num_seed_vectors=config.num_seed_vectors,
            num_cross_section_row_blocks=config.cross_row_blocks,
            num_cross_section_column_blocks=config.cross_column_blocks,
            num_temporal_layers=config.temporal_layers,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
        )
        self.parent_interaction = TargetAwareParentInteraction(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.parent_hidden_dim,
            dropout=config.dropout,
        )
        self.decoder = CausalRankingDecoder(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.decoder_hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        factor_observation_mask = derive_factor_observation_mask(batch["feature_mask"])
        # 没有任何目标观测的 episode 不应产生逐因子监督。
        target_available = batch["target_mask"].any(dim=(1, 2), keepdim=False)
        factor_observation_mask = (
            factor_observation_mask & target_available.unsqueeze(1)
        )
        # v2 明确给出满足最低观测数/比例的监督候选；v1 安全退化为可观测因子。
        supervision_mask = batch.get(
            "parent_candidate_mask", factor_observation_mask
        ) & factor_observation_mask
        market_state = batch.get("C")
        factor_features, target_feature = self.encoder(
            batch["X"],
            batch["Y"],
            market_state=market_state,
            time_padding_mask=batch["time_padding_mask"],
            asset_mask=batch["asset_mask"],
            feature_mask=batch["feature_mask"],
            target_mask=batch["target_mask"],
        )
        parent_features = self.parent_interaction(
            factor_features,
            target_feature,
            factor_mask=factor_observation_mask,
        )
        outputs = self.decoder(parent_features, return_auxiliary=True)
        return outputs, supervision_mask


def compute_losses(
    outputs: Mapping[str, Tensor],
    labels: Tensor,
    effects: Tensor,
    factor_mask: Tensor,
    effect_weight: float,
    rank_weight: float,
) -> Dict[str, Tensor]:
    """计算 proposal 的存在性、效应强度和加权 pairwise 排序目标。"""

    labels = labels.to(dtype=outputs["scores"].dtype)
    effects = effects.to(dtype=outputs["scores"].dtype)
    valid = factor_mask

    causal_values = F.binary_cross_entropy_with_logits(
        outputs["existence_logits"], labels, reduction="none"
    )
    causal_num = causal_values.masked_fill(~valid, 0.0).sum()
    causal_den = valid.sum().to(causal_num.dtype)
    causal_loss = causal_num / causal_den.clamp_min(1.0)

    parent_mask = valid & labels.bool()
    effect_errors = (outputs["effect_strength"] - effects).square()
    effect_num = effect_errors.masked_fill(~parent_mask, 0.0).sum()
    effect_den = parent_mask.sum().to(effect_num.dtype)
    effect_loss = effect_num / effect_den.clamp_min(1.0)

    tau_difference = effects.unsqueeze(2) - effects.unsqueeze(1)
    score_difference = outputs["scores"].unsqueeze(2) - outputs["scores"].unsqueeze(1)
    pair_mask = (
        valid.unsqueeze(2)
        & valid.unsqueeze(1)
        & (tau_difference > 0.0)
    )
    pair_weights = tau_difference.clamp_min(0.0).masked_fill(~pair_mask, 0.0)
    rank_num = (pair_weights * F.softplus(-score_difference)).sum()
    rank_den = pair_weights.sum()
    ranking_loss = rank_num / rank_den.clamp_min(1.0)

    total = causal_loss + effect_weight * effect_loss + rank_weight * ranking_loss
    with torch.no_grad():
        causal_correct = (
            ((outputs["existence_logits"] >= 0.0) == labels.bool()) & valid
        ).sum()
        effect_abs_num = (
            (outputs["effect_strength"] - effects).abs()
            .masked_fill(~parent_mask, 0.0)
            .sum()
        )
        pair_correct = ((score_difference > 0.0) & pair_mask).sum()
        pair_count = pair_mask.sum()
    return {
        "loss": total,
        "causal_num": causal_num.detach(),
        "causal_den": causal_den.detach(),
        "effect_num": effect_num.detach(),
        "effect_abs_num": effect_abs_num.detach(),
        "effect_den": effect_den.detach(),
        "rank_num": rank_num.detach(),
        "rank_den": rank_den.detach(),
        "causal_correct": causal_correct.detach(),
        "pair_correct": pair_correct.detach(),
        "pair_count": pair_count.detach(),
    }


def empty_totals() -> Dict[str, float]:
    return {
        name: 0.0
        for name in (
            "causal_num", "causal_den", "effect_num", "effect_abs_num",
            "effect_den", "rank_num", "rank_den", "causal_correct",
            "pair_correct", "pair_count", "batches",
        )
    }


def summarize(totals: Mapping[str, float], effect_weight: float, rank_weight: float) -> Dict[str, float]:
    causal = totals["causal_num"] / max(totals["causal_den"], 1.0)
    effect = totals["effect_num"] / max(totals["effect_den"], 1.0)
    ranking = totals["rank_num"] / max(totals["rank_den"], 1.0)
    return {
        "loss": causal + effect_weight * effect + rank_weight * ranking,
        "causal_loss": causal,
        "effect_loss": effect,
        "ranking_loss": ranking,
        "existence_accuracy": totals["causal_correct"] / max(totals["causal_den"], 1.0),
        "effect_mae": totals["effect_abs_num"] / max(totals["effect_den"], 1.0),
        "pairwise_accuracy": totals["pair_correct"] / max(totals["pair_count"], 1.0),
        "valid_factors": totals["causal_den"],
        "valid_parents": totals["effect_den"],
        "valid_pairs": totals["pair_count"],
    }


def move_batch(batch: Mapping[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def run_epoch(
    model: CausalRankModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    effect_weight: float,
    rank_weight: float,
    grad_clip: float,
    max_batches: int,
) -> Tuple[Dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    totals = empty_totals()
    steps = 0
    for batch_index, host_batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        batch = move_batch(host_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=scaler.is_enabled(),
        ):
            outputs, supervision_mask = model(batch)
            loss_items = compute_losses(
                outputs,
                batch["z"],
                batch["tau_direct"],
                supervision_mask,
                effect_weight,
                rank_weight,
            )
        if training:
            scaler.scale(loss_items["loss"]).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        for name in totals:
            if name != "batches":
                totals[name] += float(loss_items[name].item())
        totals["batches"] += 1.0
        steps += 1
    if steps == 0:
        raise RuntimeError("DataLoader 没有产生任何 batch。")
    return summarize(totals, effect_weight, rank_weight), steps


def read_split_count(dataset_directory: Path, split: str) -> int:
    with (dataset_directory / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return int(manifest["episode_counts"].get(split, 0))


def build_train_loader(args: Namespace) -> Tuple[DataLoader, CausalRankDataset]:
    primary = CausalRankDataset(
        args.train_data, "train", args.max_cached_shards, args.validate_shards
    )
    generator = torch.Generator().manual_seed(args.seed)
    dataset = primary
    sampler = None
    shuffle = True
    if args.replay_data is not None:
        replay = CausalRankDataset(
            args.replay_data, "train", args.max_cached_shards, args.validate_shards
        )
        if (primary.num_assets, primary.num_times, primary.num_factors) != (
            replay.num_assets, replay.num_times, replay.num_factors
        ):
            raise ValueError("主数据与 replay 数据的 N/T/D 必须一致。")
        dataset = ConcatDataset((primary, replay))
        primary_weight = (1.0 - args.replay_ratio) / len(primary)
        replay_weight = args.replay_ratio / len(replay)
        weights = torch.cat((
            torch.full((len(primary),), primary_weight, dtype=torch.double),
            torch.full((len(replay),), replay_weight, dtype=torch.double),
        ))
        sample_count = args.samples_per_epoch or len(primary)
        sampler = WeightedRandomSampler(
            weights, sample_count, replacement=True, generator=generator
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    return loader, primary


def build_validation_loader(args: Namespace) -> Optional[DataLoader]:
    directory = args.validation_data or args.train_data
    if read_split_count(directory, "validation") == 0:
        return None
    dataset = CausalRankDataset(
        directory, "validation", args.max_cached_shards, args.validate_shards
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )


def build_test_loader(args: Namespace) -> Optional[DataLoader]:
    """构建最终只读测试集；测试数据永远不参与参数更新或模型选择。"""

    directory = args.test_data or args.train_data
    if read_split_count(directory, "test") == 0:
        return None
    dataset = CausalRankDataset(
        directory, "test", args.max_cached_shards, args.validate_shards
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def serializable_arguments(args: Namespace) -> Dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def resolve_devices(value: str) -> Tuple[torch.device, Tuple[int, ...]]:
    """解析主设备及横截面设备；逗号分隔表示精确的多 GPU 执行。"""

    normalized = str(value).strip().lower()
    if normalized == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu"), ()
        device_ids = tuple(range(torch.cuda.device_count()))
        return torch.device(f"cuda:{device_ids[0]}"), device_ids

    parts = tuple(part.strip() for part in normalized.split(","))
    if any(not part for part in parts):
        raise ValueError("--device 包含空设备编号。")
    if len(parts) == 1 and parts[0] == "cpu":
        return torch.device("cpu"), ()

    device_ids = []
    for part in parts:
        if part.isdigit():
            device_id = int(part)
        elif part == "cuda":
            device_id = 0
        elif part.startswith("cuda:") and part[5:].isdigit():
            device_id = int(part[5:])
        else:
            raise ValueError(
                "--device 必须是 auto、cpu、N、cuda:N 或逗号分隔的 GPU 列表。"
            )
        device_ids.append(device_id)
    if not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch/CUDA 不可用。")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("--device 不能包含重复 GPU。")
    gpu_count = torch.cuda.device_count()
    if any(device_id >= gpu_count for device_id in device_ids):
        raise ValueError(
            f"请求 GPU {device_ids}，但当前仅检测到 {gpu_count} 张 GPU。"
        )
    return torch.device(f"cuda:{device_ids[0]}"), tuple(device_ids)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device, cross_section_device_ids = resolve_devices(args.device)
    use_amp = args.amp and device.type == "cuda"
    train_loader, primary_dataset = build_train_loader(args)
    validation_loader = build_validation_loader(args)
    test_loader = build_test_loader(args)
    config = ModelConfig(
        market_state_dim=primary_dataset.market_state_dim,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        num_inducing_points=args.num_inducing_points,
        num_seed_vectors=args.num_seed_vectors,
        cross_row_blocks=args.cross_row_blocks,
        cross_column_blocks=args.cross_column_blocks,
        temporal_layers=args.temporal_layers,
        max_time_steps=args.max_time_steps,
        parent_hidden_dim=args.parent_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout,
    )
    if config.max_time_steps < primary_dataset.num_times:
        raise ValueError("max_time_steps 小于数据集 T。")
    model = CausalRankModel(config).to(device)
    model.encoder.configure_cross_section_execution(
        device_ids=cross_section_device_ids,
        chunk_size=args.cross_section_chunk_size,
        use_checkpoint=args.cross_section_checkpoint,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, global_step, best_metric = 1, 0, float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["model_config"] != asdict(config):
            raise ValueError("resume checkpoint 的模型配置与当前参数不一致。")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        restore_rng_state(checkpoint["rng_state"])
        if checkpoint.get("loader_generator_state") is not None:
            train_loader.generator.set_state(checkpoint["loader_generator_state"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "training_state.json"
    history_path = args.output_dir / "history.jsonl"
    state = {
        "status": "running", "start_epoch": start_epoch,
        "target_epochs": args.epochs, "device": str(device),
        "last_completed_epoch": start_epoch - 1,
        "global_step": global_step, "best_metric": best_metric,
        "arguments": serializable_arguments(args), "model_config": asdict(config),
    }
    if args.resume is not None:
        state["last_checkpoint"] = str(args.resume)
    atomic_json(state_path, state)
    print(
        f"device={device} train={len(primary_dataset)} "
        f"validation={'none' if validation_loader is None else len(validation_loader.dataset)} "
        f"test={'none' if test_loader is None else len(test_loader.dataset)} "
        f"cross_section_devices={cross_section_device_ids or (str(device),)} "
        f"checkpoint={args.cross_section_checkpoint}",
        flush=True,
    )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, steps = run_epoch(
                model, train_loader, device, optimizer, scaler,
                args.effect_weight, args.rank_weight, args.grad_clip,
                args.max_train_batches,
            )
            global_step += steps
            validation_metrics = None
            if validation_loader is not None:
                with torch.no_grad():
                    validation_metrics, _ = run_epoch(
                        model, validation_loader, device, None, scaler,
                        args.effect_weight, args.rank_weight, 0.0,
                        args.max_validation_batches,
                    )
            selection_metric = (
                train_metrics["loss"] if validation_metrics is None
                else validation_metrics["loss"]
            )
            best_metric = min(best_metric, selection_metric)
            record = {
                "epoch": epoch, "global_step": global_step,
                "train": train_metrics, "validation": validation_metrics,
                "selection_metric": selection_metric,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 周期性更新同一个 checkpoint；最终 epoch 始终再更新一次。
            should_report = epoch % args.log_every == 0 or epoch == args.epochs
            checkpoint_path = None
            if should_report:
                checkpoint_payload = {
                    "format_version": 1, "epoch": epoch, "global_step": global_step,
                    "best_metric": best_metric, "model_config": asdict(config),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "rng_state": rng_state(),
                    "loader_generator_state": train_loader.generator.get_state(),
                    "arguments": serializable_arguments(args),
                }
                checkpoint_path = args.output_dir / "checkpoint.pt"
                atomic_checkpoint(checkpoint_path, checkpoint_payload)
            state.update({
                "status": "running", "last_completed_epoch": epoch,
                "global_step": global_step, "best_metric": best_metric,
            })
            if checkpoint_path is not None:
                state["last_checkpoint"] = str(checkpoint_path)
            atomic_json(state_path, state)
            if should_report:
                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"loss={train_metrics['loss']:.6f} "
                    f"train_acc={train_metrics['existence_accuracy']:.6f}",
                    flush=True,
                )
        # 测试仅在全部训练 epoch 完成后执行，不参与梯度、优化器更新或模型选择。
        test_metrics = None
        if test_loader is not None:
            with torch.no_grad():
                test_metrics, _ = run_epoch(
                    model, test_loader, device, None, scaler,
                    args.effect_weight, args.rank_weight, 0.0,
                    args.max_test_batches,
                )
        test_result = {
            "split": "test",
            "dataset": str(args.test_data or args.train_data),
            "checkpoint": str(args.output_dir / "checkpoint.pt"),
            "epoch": state["last_completed_epoch"],
            "global_step": global_step,
            "metrics": test_metrics,
        }
        atomic_json(args.output_dir / "test_metrics.json", test_result)
        state["test"] = test_metrics
        state["status"] = "complete"
        atomic_json(state_path, state)
        if test_metrics is None:
            print("test=none", flush=True)
        else:
            print(
                f"test_loss={test_metrics['loss']:.6f} "
                f"test_acc={test_metrics['existence_accuracy']:.6f} "
                f"test_pairwise_acc={test_metrics['pairwise_accuracy']:.6f} "
                f"test_effect_mae={test_metrics['effect_mae']:.6f}",
                flush=True,
            )
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = repr(error)
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
