"""生成带有高相关非父代理变量的 CausalRank 诊断数据。

相较 ``generate_simple_synthetic.py``，本测试只增加一种难度：候选 X 之间
不再相互独立。每个 episode 仍采用线性加性、高 SNR、无缺失目标，但在 X
的 DAG 中加入以下非父节点：

* 一个由全部真实父节点共同生成的聚合代理；
* 两个由单个真实父节点生成的近似副本代理。

聚合代理被刻意构造成与 Y 的边际相关性高于任一单独父节点，但目标结构方程
不读取任何代理。因此边际相关排序会失败，而包含全部 X 的条件线性回归可以
恢复直接父节点。这用于检验模型是否真正利用候选变量间的条件竞争关系。

输出仍严格遵循 CausalRankDataset v2 协议，每个 split 只写一个 NPZ 分片。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

try:
    from .generate_simple_synthetic import ARRAY_DESCRIPTIONS
except ImportError:
    from generate_simple_synthetic import ARRAY_DESCRIPTIONS


def parse_args() -> argparse.Namespace:
    """解析小规模条件因果诊断数据参数。"""

    parser = argparse.ArgumentParser(
        description="CausalRank correlated-proxy diagnostic data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "correlated_proxy_synthetic_diagnostic"
        ),
    )
    parser.add_argument("--num-assets", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--num-factors", type=int, default=16)
    parser.add_argument("--num-parents", type=int, default=3)
    parser.add_argument("--num-proxies", type=int, default=3)
    parser.add_argument("--proxy-noise", type=float, default=0.15)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--snr", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """检查结构维度，保证父节点和代理节点互不重叠。"""

    if args.num_assets < 8 or args.time_steps < 2:
        raise ValueError("num-assets 至少为 8，time-steps 至少为 2。")
    if args.num_parents != 3 or args.num_proxies != 3:
        raise ValueError("当前诊断机制固定使用 3 个父节点和 3 个代理节点。")
    if args.num_factors < args.num_parents + args.num_proxies + 1:
        raise ValueError("num-factors 必须容纳互不重叠的父节点、代理和普通负例。")
    if args.proxy_noise <= 0.0:
        raise ValueError("proxy-noise 必须为正数，避免完全共线。")
    if args.snr <= 0.0:
        raise ValueError("snr 必须为正数。")
    if any(
        count <= 0
        for count in (
            args.train_episodes,
            args.validation_episodes,
            args.test_episodes,
        )
    ):
        raise ValueError("三个 split 的 episode 数必须为正数。")


def rank_transform(latent: np.ndarray) -> np.ndarray:
    """逐时间、逐因子把潜在值转换成 [-1,1] 横截面秩。"""

    num_assets, time_steps, num_factors = latent.shape
    order = np.argsort(latent, axis=0, kind="stable")
    ranked = np.empty_like(latent, dtype=np.float32)
    values = np.linspace(-1.0, 1.0, num_assets, dtype=np.float32)
    for time_index in range(time_steps):
        for factor_index in range(num_factors):
            ranked[
                order[:, time_index, factor_index], time_index, factor_index
            ] = values
    return ranked


def choose_balanced_roles(
    rng: np.random.Generator,
    parent_counts: np.ndarray,
    proxy_counts: np.ndarray,
    num_parents: int,
    num_proxies: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """均衡并随机分配父/代理列，防止模型利用固定列编号。"""

    parent_order = np.lexsort((rng.random(parent_counts.size), parent_counts))
    parents = np.sort(parent_order[:num_parents]).astype(np.int64)
    parent_counts[parents] += 1

    available = np.setdiff1d(
        np.arange(parent_counts.size, dtype=np.int64), parents, assume_unique=True
    )
    proxy_order = np.lexsort((rng.random(available.size), proxy_counts[available]))
    proxies = available[proxy_order[:num_proxies]]
    proxy_counts[proxies] += 1
    return parents, proxies.astype(np.int64)


def flattened_abs_correlations(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """计算审计用的逐因子边际绝对相关性。"""

    factors = X.reshape(-1, X.shape[-1]).astype(np.float64)
    target = Y.reshape(-1).astype(np.float64)
    factors -= factors.mean(axis=0, keepdims=True)
    target -= target.mean()
    numerator = factors.T @ target
    denominator = np.sqrt(
        ((factors * factors).sum(axis=0)) * (target * target).sum()
    )
    return np.abs(numerator / np.maximum(denominator, 1e-12))


def make_episode(
    args: argparse.Namespace,
    rng: np.random.Generator,
    parent_counts: np.ndarray,
    proxy_counts: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """生成一个带显式 parent->proxy DAG 的线性目标 episode。"""

    parents, proxies = choose_balanced_roles(
        rng,
        parent_counts,
        proxy_counts,
        args.num_parents,
        args.num_proxies,
    )
    latent = rng.standard_normal(
        (args.num_assets, args.time_steps, args.num_factors)
    )

    magnitudes = rng.permutation(
        np.linspace(0.8, 1.6, args.num_parents, dtype=np.float32)
    )
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32), size=args.num_parents
    )
    parent_coefficients = magnitudes * signs

    # proxy[0] 是所有父节点的有噪声下游聚合，权重方向与目标机制对齐；它的
    # 边际相关性通常最高，但给定真实父节点后不再提供目标结构信息。
    aligned_weights = parent_coefficients / np.linalg.norm(parent_coefficients)
    latent[:, :, proxies[0]] = (
        np.einsum(
            "ntk,k->nt", latent[:, :, parents], aligned_weights, optimize=True
        )
        + args.proxy_noise
        * rng.standard_normal((args.num_assets, args.time_steps))
    )
    # 另外两个代理是首、末父节点的近似副本，形成明确的单边 DAG 边。
    for proxy, parent in zip(proxies[1:], parents[[0, -1]]):
        latent[:, :, proxy] = (
            latent[:, :, parent]
            + args.proxy_noise
            * rng.standard_normal((args.num_assets, args.time_steps))
        )

    X = rank_transform(latent)
    coefficients = np.zeros(args.num_factors, dtype=np.float32)
    coefficients[parents] = parent_coefficients
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
    q_low, q_high = np.quantile(X[:, :, 0], (0.2, 0.8))
    tau_direct[parents] = (
        np.abs(coefficients[parents]) * float(q_high - q_low) / y_std
    )

    asset_mask = np.ones((args.num_assets, args.time_steps), dtype=np.bool_)
    arrays = {
        "X": X,
        "Y": Y,
        "z": z,
        "tau_direct": tau_direct,
        "asset_mask": asset_mask,
        "feature_mask": np.ones(X.shape, dtype=np.bool_),
        "parent_candidate_mask": np.ones(args.num_factors, dtype=np.bool_),
        "target_mask": asset_mask.copy(),
        "time_padding_mask": np.zeros(args.time_steps, dtype=np.bool_),
    }

    correlations = flattened_abs_correlations(X, Y)
    edges = [
        [int(parent), int(proxies[0]), float(weight)]
        for parent, weight in zip(parents, aligned_weights)
    ]
    edges.extend(
        [int(parent), int(proxy), 1.0]
        for proxy, parent in zip(proxies[1:], parents[[0, -1]])
    )
    audit = {
        "parents": parents.tolist(),
        "proxies": proxies.tolist(),
        "parent_coefficients": parent_coefficients.tolist(),
        "feature_dag_edges": edges,
        "parent_abs_correlations": correlations[parents].tolist(),
        "proxy_abs_correlations": correlations[proxies].tolist(),
        "aggregate_proxy_exceeds_all_parents": bool(
            correlations[proxies[0]] > correlations[parents].max()
        ),
        "noise_scale": noise_scale,
    }
    return arrays, audit


def generate_split(
    args: argparse.Namespace,
    split: str,
    episode_count: int,
    seed: np.random.SeedSequence,
) -> Tuple[str, List[int], List[int], List[Dict[str, object]]]:
    """生成一个split并合并到一个压缩分片。"""

    rng = np.random.default_rng(seed)
    parent_counts = np.zeros(args.num_factors, dtype=np.int64)
    proxy_counts = np.zeros(args.num_factors, dtype=np.int64)
    episodes: List[Dict[str, np.ndarray]] = []
    audit: List[Dict[str, object]] = []
    for _ in range(episode_count):
        arrays, episode_audit = make_episode(
            args, rng, parent_counts, proxy_counts
        )
        episodes.append(arrays)
        audit.append(episode_audit)

    split_directory = args.output_dir / split
    split_directory.mkdir(parents=True)
    shard_path = split_directory / "shard_00000.npz"
    np.savez_compressed(
        shard_path,
        **{
            name: np.stack([episode[name] for episode in episodes], axis=0)
            for name in ARRAY_DESCRIPTIONS
        },
    )
    return (
        str(shard_path.relative_to(args.output_dir)),
        parent_counts.tolist(),
        proxy_counts.tolist(),
        audit,
    )


def main() -> None:
    """写入三个split，并在所有分片完成后原子发布manifest。"""

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
    role_counts: Dict[str, Dict[str, List[int]]] = {}
    audits: Dict[str, List[Dict[str, object]]] = {}
    for (split, episode_count), seed in zip(counts.items(), split_seeds):
        shard, parent_counts, proxy_counts, audit = generate_split(
            args, split, episode_count, seed
        )
        shards.append(shard)
        role_counts[split] = {
            "parents": parent_counts,
            "proxies": proxy_counts,
        }
        audits[split] = audit

    manifest = {
        "format_version": 2,
        "dataset_name": "CausalRank correlated-proxy linear diagnostic episodes",
        "source_type": "correlated_proxy_synthetic_diagnostic",
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
            "X": "root factors plus parent-to-proxy DAG edges, then cross-sectional ranks",
            "Y": "standardized linear additive target reading parents only",
            "missingness": "none",
            "snr": args.snr,
            "proxy_noise": args.proxy_noise,
            "role_selection_counts": role_counts,
            "episode_audit": audits,
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
