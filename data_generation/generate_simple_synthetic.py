"""生成用于诊断 CausalRank 训练链路的最简合成面板。

这个数据集不是论文的最终数据生成机制，而是一项可证伪的工程测试：每个
episode 的候选因子彼此独立，目标由三个随机位置的父节点线性加性生成，
不存在缺失、因子间 DAG、非线性、交互或异方差。若完整模型无法在该数据上
获得接近满分的训练与测试排序，问题应优先归因于表示或训练实现，而不是复杂
SCM 的统计难度。

输出严格遵循 CausalRankDataset 的 v2 九字段协议。为避免制造大量小文件，
train、validation、test 各只保存一个压缩 NPZ 分片。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np


ARRAY_DESCRIPTIONS: Mapping[str, str] = {
    "X": "float32 [E,N,T,D]",
    "Y": "float32 [E,N,T]",
    "z": "bool [E,D]",
    "tau_direct": "float32 [E,D]",
    "asset_mask": "bool [E,N,T]",
    "feature_mask": "bool [E,N,T,D]",
    "parent_candidate_mask": "bool [E,D]",
    "target_mask": "bool [E,N,T]",
    "time_padding_mask": "bool [E,T]",
}


def parse_args() -> argparse.Namespace:
    """解析诊断数据规模；默认值刻意保持小而统计信号充分。"""

    parser = argparse.ArgumentParser(description="CausalRank simple diagnostic data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simple_synthetic_diagnostic",
    )
    parser.add_argument("--num-assets", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--num-factors", type=int, default=16)
    parser.add_argument("--num-parents", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--snr", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在创建目录前拒绝会破坏诊断含义或 Dataset 协议的参数。"""

    if args.num_assets < 8 or args.time_steps < 2:
        raise ValueError("num-assets 至少为 8，time-steps 至少为 2。")
    if args.num_factors < 2:
        raise ValueError("num-factors 至少为 2。")
    if not 1 <= args.num_parents < args.num_factors:
        raise ValueError("num-parents 必须满足 1 <= K < D。")
    if any(
        count <= 0
        for count in (
            args.train_episodes,
            args.validation_episodes,
            args.test_episodes,
        )
    ):
        raise ValueError("三个 split 的 episode 数都必须为正数。")
    if args.snr <= 0.0:
        raise ValueError("snr 必须为正数。")


def cross_sectional_ranks(
    rng: np.random.Generator,
    num_assets: int,
    time_steps: int,
    num_factors: int,
) -> np.ndarray:
    """生成相互独立、每期截面严格均匀分布在 [-1,1] 的候选因子。"""

    latent = rng.standard_normal((num_assets, time_steps, num_factors))
    order = np.argsort(latent, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    rank_values = np.linspace(-1.0, 1.0, num_assets, dtype=np.float32)
    # 每个 (time, factor) 截面都是 rank_values 的独立随机排列。
    for time_index in range(time_steps):
        for factor_index in range(num_factors):
            ranks[order[:, time_index, factor_index], time_index, factor_index] = (
                rank_values
            )
    return ranks


def choose_balanced_parents(
    rng: np.random.Generator,
    selection_counts: np.ndarray,
    num_parents: int,
) -> np.ndarray:
    """优先选历史出现较少的列，并随机打破并列，防止位置先验泄漏。"""

    tie_breaker = rng.random(selection_counts.size)
    order = np.lexsort((tie_breaker, selection_counts))
    parents = np.sort(order[:num_parents]).astype(np.int64)
    selection_counts[parents] += 1
    return parents


def make_episode(
    args: argparse.Namespace,
    rng: np.random.Generator,
    selection_counts: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """生成一个独立线性 episode 及少量用于 manifest 审计的统计量。"""

    X = cross_sectional_ranks(
        rng, args.num_assets, args.time_steps, args.num_factors
    )
    parents = choose_balanced_parents(rng, selection_counts, args.num_parents)

    # 预设分离良好的强度，再随机分配给父节点并随机取符号。这样父节点集合与
    # 父节点内部的真实强弱顺序都可从 X-Y 关系中稳定辨认。
    magnitudes = np.linspace(0.8, 1.6, args.num_parents, dtype=np.float32)
    magnitudes = rng.permutation(magnitudes)
    coefficients = np.zeros(args.num_factors, dtype=np.float32)
    coefficients[parents] = magnitudes * rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32), size=args.num_parents
    )

    signal = np.einsum("ntd,d->nt", X, coefficients, optimize=True)
    signal_scale = max(float(signal.std()), 1e-6)
    noise_scale = signal_scale / np.sqrt(args.snr)
    y_raw = signal + rng.normal(0.0, noise_scale, signal.shape)
    y_mean = float(y_raw.mean())
    y_std = max(float(y_raw.std()), 1e-6)
    Y = ((y_raw - y_mean) / y_std).astype(np.float32)

    z = np.zeros(args.num_factors, dtype=np.bool_)
    z[parents] = True
    tau_direct = np.zeros(args.num_factors, dtype=np.float32)
    # 所有截面共享相同的秩网格，因此 q20->q80 的干预跨度对所有列相同。
    q_low, q_high = np.quantile(X[:, :, 0], (0.2, 0.8))
    tau_direct[parents] = (
        np.abs(coefficients[parents]) * float(q_high - q_low) / y_std
    )

    asset_mask = np.ones((args.num_assets, args.time_steps), dtype=np.bool_)
    arrays = {
        "X": X.astype(np.float32, copy=False),
        "Y": Y,
        "z": z,
        "tau_direct": tau_direct,
        "asset_mask": asset_mask,
        "feature_mask": np.ones(X.shape, dtype=np.bool_),
        "parent_candidate_mask": np.ones(args.num_factors, dtype=np.bool_),
        "target_mask": asset_mask.copy(),
        "time_padding_mask": np.zeros(args.time_steps, dtype=np.bool_),
    }
    audit = {
        "parents": parents.tolist(),
        "coefficients": coefficients[parents].tolist(),
        "noise_scale": noise_scale,
    }
    return arrays, audit


def generate_split(
    args: argparse.Namespace,
    split: str,
    episode_count: int,
    seed: np.random.SeedSequence,
) -> Tuple[str, List[int], List[Dict[str, object]]]:
    """生成一个 split 并合并为单一分片。"""

    rng = np.random.default_rng(seed)
    selection_counts = np.zeros(args.num_factors, dtype=np.int64)
    episodes: List[Dict[str, np.ndarray]] = []
    audit: List[Dict[str, object]] = []
    for _ in range(episode_count):
        arrays, episode_audit = make_episode(args, rng, selection_counts)
        episodes.append(arrays)
        audit.append(episode_audit)

    split_directory = args.output_dir / split
    split_directory.mkdir(parents=True)
    shard_path = split_directory / "shard_00000.npz"
    stacked = {
        name: np.stack([episode[name] for episode in episodes], axis=0)
        for name in ARRAY_DESCRIPTIONS
    }
    np.savez_compressed(shard_path, **stacked)
    return str(shard_path.relative_to(args.output_dir)), selection_counts.tolist(), audit


def main() -> None:
    """生成三个独立 split，并以原子方式最后写入完成标志 manifest。"""

    args = parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"输出目录非空：{args.output_dir}；请更换目录，避免覆盖已有数据。"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
        "test": args.test_episodes,
    }
    split_seeds = np.random.SeedSequence(args.seed).spawn(len(counts))
    shards: List[str] = []
    parent_counts: Dict[str, List[int]] = {}
    audit_summary: Dict[str, List[Dict[str, object]]] = {}
    for (split, episode_count), seed in zip(counts.items(), split_seeds):
        shard, split_parent_counts, audit = generate_split(
            args, split, episode_count, seed
        )
        shards.append(shard)
        parent_counts[split] = split_parent_counts
        audit_summary[split] = audit

    manifest = {
        "format_version": 2,
        "dataset_name": "CausalRank simple linear diagnostic episodes",
        "source_type": "simple_synthetic_diagnostic",
        "feature_names": [f"x{index:02d}" for index in range(args.num_factors)],
        "dimensions": {
            "N": args.num_assets,
            "T": args.time_steps,
            "D": args.num_factors,
        },
        "market_state_dim": 0,
        "contains_C": False,
        "episode_counts": counts,
        "episodes_per_shard": "one shard per split",
        "shards": shards,
        "training_arrays": dict(ARRAY_DESCRIPTIONS),
        "generation": {
            "X": "independent cross-sectional ranks; no temporal or cross-factor edges",
            "Y": "standardized linear additive target with independent Gaussian noise",
            "missingness": "none",
            "num_parents": args.num_parents,
            "snr": args.snr,
            "coefficient_magnitudes": np.linspace(
                0.8, 1.6, args.num_parents
            ).tolist(),
            "parent_selection_counts": parent_counts,
            "episode_audit": audit_summary,
        },
        "tau_semantics": "absolute q20-to-q80 controlled direct linear effect after Y standardization",
        "seed": args.seed,
        "generation_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    temporary_manifest = args.output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(args.output_dir / "manifest.json")
    print(
        f"generated {sum(counts.values())} episodes, 3 shards at {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
