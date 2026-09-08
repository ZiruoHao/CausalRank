#!/usr/bin/env python3
"""生成由 GKX 真实分布校准的全合成动态 SCM episode。

全合成与半合成数据使用相同的 N、T、D 和训练数组接口。程序只从真实 GKX
训练期估计以下分布摘要：股票月覆盖率、各特征观测率、逐特征一阶时序相关
以及特征间截面相关。之后的 X 和 Y 都由已知结构方程重新生成，不复制真实
股票记录。

X 的完整因果结构
-----------------
每个 episode 随机抽取一个特征拓扑序，生成稀疏同时刻 DAG，并在分片外保存
``feature_adjacency`` 和 ``feature_scm_coefficients``；每个特征还具有保存于
``feature_ar_coefficients`` 的自身滞后边。DAG 的边抽样会偏向真实数据中高
相关的特征对，AR 系数、缺失率和资产覆盖率也由真实训练期校准。生成完整
潜在 X 后，再施加股票上市区间和具有持续性的特征缺失过程。

Y 与监督
--------
目标机制与半合成脚本共用同一实现，输出 z 和 q20→q80 受控直接效应
tau_direct。全合成 X 图已知，因此以后可以扩展总效应 tau_total；当前训练
分片只保存论文主任务需要的直接效应，不把祖先总效应与直接父节点混用。
父节点只从达到最低观测数和观测率的特征中选择，并在每个数据 split 内按
历史入选次数进行均衡抽样；训练分片保存 ``parent_candidate_mask``。

当前 GKX DataShare 没有宏观状态 C，因此本程序固定 K=0、不保存 C 数组；
训练时使用 ``market_state_dim=0`` 和 ``C=None``。
"""

from __future__ import annotations  # 推迟类型标注求值，兼容 Python 3.9。

# argparse 管理命令行；json 写逐 episode 元数据；math 提供 SCM 数值运算。
import argparse
import gzip
import json
import math
import os
# 相邻校准窗口会复用月份，LRU 缓存减少重复解压。
from functools import lru_cache
# Path 用于构造不依赖当前工作目录的项目路径。
from pathlib import Path
# 类型标注明确校准字典和输出张量容器。
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# NumPy 负责 SCM、掩码、张量及 NPZ 数据处理。
import numpy as np

# 复用半合成脚本的缓存、秩变换、Y 机制和统一分片协议，避免两套标签定义漂移。
from generate_semi_synthetic import (
    CalibrationAccumulator,
    MIN_QUANTILE_OBSERVATIONS,
    atomic_json,
    build_month_cache,
    feature_columns,
    find_project_root,
    split_calendar,
    stack_and_save,
    synthesize_target,
    valid_window_starts,
    rank_transform,
    write_dataset_readme,
)


# 输出新增 parent_candidate_mask，与半合成协议同步升级到版本 2。
FORMAT_VERSION = 2


def parse_args() -> argparse.Namespace:
    """定义全合成规模、SCM 稀疏度、校准方式和保存位置。"""

    # 自动定位项目根目录，因此移动 data_generation 文件夹后仍能找到真实数据。
    project_root = find_project_root()
    script_directory = Path(__file__).resolve().parent
    # 默认引用项目内的 GKX 原始数据与两个脚本共享的逐月缓存。
    default_data = project_root / "data" / "GKX Characteristics Data"
    parser = argparse.ArgumentParser(
        description="生成由 GKX 训练期分布校准的全合成动态 SCM episode。"
    )
    # 真实数据只用于校准；输出中不会复制这些股票记录。
    parser.add_argument("--input-csv", type=Path, default=default_data / "datashare.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=script_directory / "full_synthetic",
        help="输出目录；相对路径按运行命令时的当前目录解析。",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=default_data / "monthly_npz_cache",
        help="与半合成生成器共享的 GKX 逐月缓存根目录。",
    )
    parser.add_argument(
        "--calibration", type=Path, default=None,
        help="可选的半合成 calibration.json（也兼容旧 NPZ）；省略时自动估计。",
    )
    # 与半合成数据保持相同的 N/T、split 数量和分片大小。
    parser.add_argument("--num-assets", type=int, default=256)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--train-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--episodes-per-shard", type=int, default=8)
    # Y 的直接父节点数、目标信噪比和困难负样本判定阈值。
    parser.add_argument("--min-parents", type=int, default=1)
    parser.add_argument("--max-parents", type=int, default=5)
    parser.add_argument(
        "--min-parent-observations", type=int, default=16,
        help="父节点候选特征在 episode 内至少需要的有效观测数。",
    )
    parser.add_argument(
        "--min-parent-observation-rate", type=float, default=0.50,
        help="父节点候选特征相对有效股票月的最低观测率。",
    )
    parser.add_argument("--snr-low", type=float, default=0.5)
    parser.add_argument("--snr-high", type=float, default=5.0)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.50)
    # 特征动态 SCM、burn-in 与缺失 Markov 过程的控制参数。
    parser.add_argument("--expected-indegree", type=float, default=2.0)
    parser.add_argument("--max-indegree", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--missing-persistence", type=float, default=0.90)
    parser.add_argument(
        "--asset-observation-rate", type=float, default=None,
        help="覆盖校准得到的股票平均有效月份比例。",
    )
    # 未提供 calibration.json 时，用多少真实训练窗口自行估计分布。
    parser.add_argument("--calibration-windows", type=int, default=32)
    parser.add_argument("--calibration-train-fraction", type=float, default=0.70)
    parser.add_argument("--min-asset-coverage", type=float, default=0.80)
    # 全局复现 seed、CSV chunk 大小和烟雾测试截断开关。
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument(
        "--max-source-rows", type=int, default=0,
        help="校准时最多读取的 GKX 行数；0 表示全部，仅用于测试时截断。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, num_features: int) -> None:
    """在扫描真实数据或生成 SCM 前检查全部范围约束。"""

    if not args.input_csv.is_file():
        raise FileNotFoundError(f"GKX CSV does not exist: {args.input_csv}")
    for name in (
        "num_assets", "time_steps", "episodes_per_shard", "max_indegree",
        "calibration_windows", "chunksize",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("train_episodes", "validation_episodes", "test_episodes"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 1 <= args.min_parents <= args.max_parents <= num_features:
        raise ValueError("parent bounds must satisfy 1 <= min <= max <= D")
    if not MIN_QUANTILE_OBSERVATIONS <= args.min_parent_observations <= args.num_assets * args.time_steps:
        raise ValueError(
            f"--min-parent-observations must be in [{MIN_QUANTILE_OBSERVATIONS}, "
            "num-assets * time-steps]"
        )
    if not 0.0 < args.min_parent_observation_rate <= 1.0:
        raise ValueError("--min-parent-observation-rate must be in (0, 1]")
    if not 0.0 < args.snr_low <= args.snr_high:
        raise ValueError("SNR bounds must satisfy 0 < low <= high")
    if args.expected_indegree < 0.0:
        raise ValueError("--expected-indegree cannot be negative")
    if args.burn_in < 0:
        raise ValueError("--burn-in cannot be negative")
    if not 0.0 <= args.missing_persistence < 1.0:
        raise ValueError("--missing-persistence must be in [0, 1)")
    if args.asset_observation_rate is not None and not 0.0 < args.asset_observation_rate <= 1.0:
        raise ValueError("--asset-observation-rate must be in (0, 1]")
    if not 0.0 < args.calibration_train_fraction <= 1.0:
        raise ValueError("--calibration-train-fraction must be in (0, 1]")
    if not 0.0 < args.min_asset_coverage <= 1.0:
        raise ValueError("--min-asset-coverage must be in (0, 1]")


def read_calibration(path: Path, expected_features: Sequence[str]) -> Dict[str, np.ndarray]:
    """读取校准 JSON（兼容旧 NPZ），检查字段及 94 个特征的顺序。"""

    if not path.is_file():
        raise FileNotFoundError(f"calibration file does not exist: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = {
            name: np.asarray(value)
            for name, value in payload.items()
            if name != "format_version"
        }
    else:
        # 保留兼容入口，便于复用重构前已经生成的 calibration.npz。
        with np.load(path, allow_pickle=False) as data:
            result = {name: data[name].copy() for name in data.files}
    required = {
        "feature_names", "feature_observation_rate", "lag1_correlation",
        "cross_feature_correlation", "asset_observation_rate",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"calibration is missing fields: {sorted(missing)}")
    names = result["feature_names"].astype(str).tolist()
    if names != list(expected_features):
        raise ValueError("calibration feature order differs from GKX CSV")
    for name in required.difference({"feature_names"}):
        result[name] = np.asarray(result[name], dtype=np.float32)
    return result


def write_calibration_snapshot(path: Path, calibration: Mapping[str, np.ndarray]) -> None:
    """把实际使用的校准量复制为输出目录内的可读 JSON 快照。"""

    payload: Dict[str, Any] = {"format_version": FORMAT_VERSION}
    for name, value in calibration.items():
        array = np.asarray(value)
        payload[name] = array.item() if array.ndim == 0 else array.tolist()
    atomic_json(path, payload)


def build_reference_calibration(
    args: argparse.Namespace,
    features: Sequence[str],
    destination: Path,
) -> Dict[str, np.ndarray]:
    """仅从 GKX 训练期的对齐窗口估计 X 和掩码分布。

    如果没有传入半合成 ``calibration.json``，脚本会走此分支。真实数据只
    用于计算分布摘要，不会作为全合成 episode 的 X 被保存。
    """

    # 复用半合成脚本的逐月缓存协议，确保两类数据看到相同的特征顺序。
    cache_path, cache_manifest = build_month_cache(
        args.input_csv, args.cache_dir, features, args.chunksize, args.max_source_rows
    )
    all_dates = np.asarray(cache_manifest["dates"], dtype=np.int64)
    train_dates = split_calendar(all_dates, args.calibration_train_fraction, 0.0)["train"]
    starts = valid_window_starts(train_dates, args.time_steps)
    if starts.size == 0:
        raise ValueError(
            f"GKX training period has no contiguous {args.time_steps}-month calibration window"
        )

    @lru_cache(maxsize=max(2 * args.time_steps, 64))
    def load_month(date_value: int) -> Tuple[np.ndarray, np.ndarray]:
        """读取并复制一个月份的数据，使 NPZ 句柄可以立即关闭。"""

        with np.load(cache_path / f"month_{date_value}.npz", allow_pickle=False) as data:
            return data["stock_ids"].copy(), data["X_obs"].copy()

    # 校准使用独立子随机流，不消耗后续训练 episode 的随机序列。
    rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0])
    accumulator = CalibrationAccumulator(len(features))
    successful = 0
    attempts = 0
    while successful < args.calibration_windows:
        attempts += 1
        if attempts > args.calibration_windows * 20:
            raise RuntimeError("could not sample enough reference windows with eligible stocks")
        start = int(rng.choice(starts))
        dates = train_dates[start : start + args.time_steps]
        month_data = [load_month(int(date)) for date in dates]
        all_ids = np.concatenate([item[0] for item in month_data])
        # 股票资格只依赖覆盖率；不会依据未来的目标父节点选择样本。
        unique_ids, counts = np.unique(all_ids, return_counts=True)
        minimum_count = int(math.ceil(args.min_asset_coverage * args.time_steps))
        eligible = unique_ids[counts >= minimum_count]
        if eligible.size == 0:
            continue
        selected_count = min(args.num_assets, eligible.size)
        selected = np.sort(rng.choice(eligible, size=selected_count, replace=False))
        X_obs = np.full(
            (args.num_assets, args.time_steps, len(features)), np.nan, dtype=np.float32
        )
        asset_mask = np.zeros((args.num_assets, args.time_steps), dtype=bool)
        for time_index, (month_ids, month_values) in enumerate(month_data):
            positions = np.searchsorted(month_ids, selected)
            present = positions < month_ids.size
            present[present] &= month_ids[positions[present]] == selected[present]
            target_rows = np.flatnonzero(present)
            X_obs[target_rows, time_index, :] = month_values[positions[present]]
            asset_mask[target_rows, time_index] = True
        X, mask = rank_transform(X_obs, asset_mask)
        accumulator.update(X, asset_mask, mask, rng)
        successful += 1

    accumulator.save(destination, features)
    return read_calibration(destination, features)


def generate_asset_mask(
    num_assets: int,
    time_steps: int,
    mean_observation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """为每只合成股票生成窗口内唯一且连续的上市区间。

    区间长度由以真实覆盖率为均值的 Beta 分布抽取，因而可模拟窗口中途
    上市或退市；asset_mask 之外的 X/Y 均不会参与训练。
    """

    # concentration 越大，各股票覆盖率越集中在真实均值附近。
    mean_rate = float(np.clip(mean_observation_rate, 0.05, 1.0))
    concentration = 24.0
    alpha = max(mean_rate * concentration, 0.2)
    beta = max((1.0 - mean_rate) * concentration, 0.2)
    coverage = rng.beta(alpha, beta, size=num_assets)
    lengths = np.clip(np.rint(coverage * time_steps).astype(int), 1, time_steps)
    mask = np.zeros((num_assets, time_steps), dtype=bool)
    for asset, length in enumerate(lengths):
        start = int(rng.integers(0, time_steps - int(length) + 1))
        mask[asset, start : start + int(length)] = True
    return mask


def sample_sparse_scm(
    reference_correlation: np.ndarray,
    lag_correlation: np.ndarray,
    expected_indegree: float,
    max_indegree: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """采样稀疏有向无环 SEM，并偏向真实数据中高相关的特征对。

    adjacency[parent, child]=True。先抽拓扑序再只从前驱中抽父节点，因此
    无需事后删环；每条边、AR 系数和节点非线性都会随 episode 保存。
    """

    num_features = reference_correlation.shape[0]
    # 随机拓扑序让不同 episode 的变量角色变化，避免固定列位置泄露标签。
    topological_order = rng.permutation(num_features).astype(np.int64)
    adjacency = np.zeros((num_features, num_features), dtype=bool)
    coefficients = np.zeros((num_features, num_features), dtype=np.float32)
    node_nonlinearity = rng.integers(0, 3, size=num_features, dtype=np.int8)
    for position, child in enumerate(topological_order):
        if position == 0:
            continue
        candidates = topological_order[:position]
        indegree = min(position, max_indegree, int(rng.poisson(expected_indegree)))
        if indegree == 0:
            continue
        # 相关性只提高边被抽中的概率，不直接当作因果方向或边系数。
        weights = 0.05 + np.abs(reference_correlation[child, candidates])
        probabilities = weights / weights.sum()
        parents = rng.choice(candidates, size=indegree, replace=False, p=probabilities)
        for parent in parents:
            empirical_sign = np.sign(reference_correlation[child, parent])
            sign = empirical_sign if empirical_sign != 0 else rng.choice([-1.0, 1.0])
            coefficient = sign * rng.uniform(0.15, 0.65) / math.sqrt(indegree)
            adjacency[parent, child] = True
            coefficients[parent, child] = coefficient
    # 用真实 lag-1 相关作为 AR 中心，并加 episode 级扰动以扩大机制覆盖。
    ar_coefficients = np.clip(
        np.nan_to_num(lag_correlation, nan=0.70) + rng.normal(0.0, 0.04, num_features),
        -0.20,
        0.98,
    ).astype(np.float32)
    return adjacency, coefficients, ar_coefficients, topological_order, node_nonlinearity


def generate_latent_x(
    num_assets: int,
    time_steps: int,
    burn_in: int,
    adjacency: np.ndarray,
    coefficients: np.ndarray,
    ar_coefficients: np.ndarray,
    topological_order: np.ndarray,
    node_nonlinearity: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """按拓扑序生成完整、尚未施加缺失的动态潜在 X。

    每个特征由自身上一期、同一期 DAG 父节点和独立 Gaussian 创新组成。
    burn-in 不写入输出，用于减小随机初值造成的非平稳影响。
    """

    num_features = adjacency.shape[0]
    # previous 保存 t-1；current 按拓扑序逐列填充同一期结构值。
    previous = rng.standard_normal((num_assets, num_features)).astype(np.float64)
    saved = np.empty((num_assets, time_steps, num_features), dtype=np.float32)
    for step in range(burn_in + time_steps):
        innovation = rng.standard_normal((num_assets, num_features))
        current = np.zeros_like(previous)
        for child in topological_order:
            parents = np.flatnonzero(adjacency[:, child])
            structural = np.zeros(num_assets, dtype=np.float64)
            if parents.size:
                structural = current[:, parents] @ coefficients[parents, child].astype(np.float64)
                code = int(node_nonlinearity[child])
                if code == 1:
                    structural = np.tanh(structural)
                elif code == 2:
                    structural = np.sign(structural) * np.sqrt(np.abs(structural) + 1e-8)
            rho = float(ar_coefficients[child])
            innovation_scale = math.sqrt(max(1.0 - rho * rho, 1e-4))
            current[:, child] = (
                rho * previous[:, child] + structural + innovation_scale * innovation[:, child]
            )
        previous = current
        if step >= burn_in:
            saved[:, step - burn_in, :] = current.astype(np.float32)
    return saved


def cross_sectional_rank_dense(latent: np.ndarray, asset_mask: np.ndarray) -> np.ndarray:
    """在每个合成月份的有效股票截面内把潜在 X 映射到 [-1,1]。"""

    output = np.zeros_like(latent, dtype=np.float32)
    _, num_times, num_features = latent.shape
    for time_index in range(num_times):
        active = np.flatnonzero(asset_mask[:, time_index])
        count = active.size
        if count <= 1:
            continue
        for feature_index in range(num_features):
            values = latent[active, time_index, feature_index]
            order = np.argsort(values, kind="stable")
            ranks = np.empty(count, dtype=np.float32)
            ranks[order] = np.arange(count, dtype=np.float32)
            output[active, time_index, feature_index] = 2.0 * ranks / (count - 1) - 1.0
    return output


def generate_feature_mask(
    asset_mask: np.ndarray,
    observation_rate: np.ndarray,
    missing_persistence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """按真实特征观测率生成具有时间持续性的 feature_mask。

    对每个股票—特征使用二状态 Markov 缺失过程，并在最后与 asset_mask
    相交，保证未上市股票不可能出现被观测的特征。
    """

    num_assets, num_times = asset_mask.shape
    num_features = observation_rate.size
    # 极端概率会造成永远观测/永远缺失，因此留出很小的数值边界。
    observation_rate = np.clip(observation_rate.astype(np.float64), 0.01, 0.999)
    missing_rate = 1.0 - observation_rate
    missing_after_observed = missing_rate * (1.0 - missing_persistence) / observation_rate
    missing_after_observed = np.clip(missing_after_observed, 0.0, 1.0)
    observed = rng.random((num_assets, num_features)) < observation_rate
    mask = np.empty((num_assets, num_times, num_features), dtype=bool)
    mask[:, 0, :] = observed
    for time_index in range(1, num_times):
        draw = rng.random((num_assets, num_features))
        remain_missing = (~observed) & (draw < missing_persistence)
        become_missing = observed & (draw < missing_after_observed)
        observed = ~(remain_missing | become_missing)
        mask[:, time_index, :] = observed
    return mask & asset_mask[:, :, None]


def make_episode(
    args: argparse.Namespace,
    calibration: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    parent_selection_counts: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """生成一个完整全合成 episode 及其可序列化元数据。

    顺序为 A → 特征 DAG → 完整潜在 X → 截面秩 → M → 已知目标方程。
    输出还保存 SCM 图和参数，使 X 的因果结构能够被复核或用于辅助监督。
    """

    feature_observation_rate = calibration["feature_observation_rate"]
    lag_correlation = calibration["lag1_correlation"]
    reference_correlation = calibration["cross_feature_correlation"]
    calibrated_asset_rate = float(calibration["asset_observation_rate"])
    asset_rate = (
        calibrated_asset_rate
        if args.asset_observation_rate is None
        else args.asset_observation_rate
    )
    # A 表示股票月是否存在；后续所有特征和目标掩码都从 A 派生不能越界。
    asset_mask = generate_asset_mask(args.num_assets, args.time_steps, asset_rate, rng)
    # 每个 episode 独立抽取一个已知的稀疏动态特征 SCM。
    adjacency, scm_coefficients, ar_coefficients, order, node_nonlinearity = sample_sparse_scm(
        reference_correlation, lag_correlation,
        args.expected_indegree, args.max_indegree, rng,
    )
    latent = generate_latent_x(
        args.num_assets, args.time_steps, args.burn_in,
        adjacency, scm_coefficients, ar_coefficients, order, node_nonlinearity, rng,
    )
    # 先生成完整潜在值，再按真实数据流程做月度截面秩和缺失机制。
    complete_rank_x = cross_sectional_rank_dense(latent, asset_mask)
    feature_mask = generate_feature_mask(
        asset_mask, feature_observation_rate, args.missing_persistence, rng
    )
    X = np.where(feature_mask, complete_rank_x, 0.0).astype(np.float32)
    # 与半合成数据共用目标生成器，确保 z/tau 的含义完全一致。
    target_arrays, target_metadata = synthesize_target(
        X, asset_mask, feature_mask, rng,
        args.min_parents, args.max_parents,
        args.min_parent_observations,
        args.min_parent_observation_rate,
        parent_selection_counts,
        args.snr_low, args.snr_high,
        args.hard_negative_threshold,
    )
    arrays: Dict[str, np.ndarray] = {
        "X": X,
        "Y": target_arrays["Y"],
        "z": target_arrays["z"],
        "tau_direct": target_arrays["tau_direct"],
        "asset_mask": asset_mask,
        "feature_mask": feature_mask,
        "parent_candidate_mask": target_arrays["parent_candidate_mask"],
        "target_mask": asset_mask.copy(),
        "time_padding_mask": ~asset_mask.any(axis=0),
    }
    edge_indices = np.argwhere(adjacency)
    metadata = {
        "asset_observation_rate": float(asset_mask.mean()),
        "feature_observation_rate_given_asset": float(
            feature_mask.sum() / max(asset_mask.sum() * feature_mask.shape[-1], 1)
        ),
        "feature_edge_count": int(adjacency.sum()),
        "feature_expected_indegree": args.expected_indegree,
        "burn_in": args.burn_in,
        # 图、目标机制和干预端点只用于生成复现/审计，不进入训练 NPZ。
        "feature_scm_edges": [
            [int(parent), int(child), float(scm_coefficients[parent, child])]
            for parent, child in edge_indices
        ],
        "feature_ar_coefficients": ar_coefficients.tolist(),
        "topological_order": order.astype(np.int16).tolist(),
        "node_nonlinearity": node_nonlinearity.tolist(),
        "q_low": target_arrays["q_low"].tolist(),
        "q_high": target_arrays["q_high"].tolist(),
        "parent_coefficients": target_arrays["parent_coefficients"].tolist(),
        "transform_codes": target_arrays["transform_codes"].tolist(),
        "transform_thresholds": target_arrays["transform_thresholds"].tolist(),
        "hard_negative_indices": np.flatnonzero(
            target_arrays["hard_negative_mask"]
        ).tolist(),
        **target_metadata,
    }
    return arrays, metadata


def main() -> None:
    """执行校准、全合成 episode 生成、分片和 manifest 保存。"""

    # 从真实 GKX 表头固定 D=94 和特征顺序，再检查全部参数。
    args = parse_args()
    # 统一转为绝对路径，避免相对输出目录再次落入意外的嵌套路径。
    args.input_csv = args.input_csv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.calibration is not None:
        args.calibration = args.calibration.expanduser().resolve()
    print(f"Input CSV: {args.input_csv}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    features = feature_columns(args.input_csv)
    validate_args(args, len(features))
    # 拒绝覆盖非空目录，避免新旧 seed 或协议的数据分片混在一起。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; choose a new directory"
        )

    # 优先复用半合成校准；输出目录始终保留本次实际使用的 JSON 快照。
    calibration_snapshot = args.output_dir / "reference_calibration.json"
    if args.calibration is not None:
        calibration_source = args.calibration.resolve()
        calibration = read_calibration(calibration_source, features)
        write_calibration_snapshot(calibration_snapshot, calibration)
        calibration_origin = "external"
    else:
        calibration_source = None
        calibration = build_reference_calibration(args, features, calibration_snapshot)
        calibration_origin = "sampled_from_gkx_training_period"

    requested = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
        "test": args.test_episodes,
    }
    # 三个 split 使用独立且可由根 seed 完全复现的随机流。
    root_seed = np.random.SeedSequence(args.seed)
    split_seed_sequences = root_seed.spawn(len(requested))
    metadata_path = args.output_dir / "generation_metadata.jsonl.gz"
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    shard_files: List[str] = []
    total_episodes = 0
    parent_selection_counts_by_split: Dict[str, List[int]] = {}

    with gzip.open(metadata_temporary, "wt", encoding="utf-8", newline="\n") as metadata_file:
        for (split, episode_count), split_seed in zip(requested.items(), split_seed_sequences):
            split_directory = args.output_dir / split
            split_directory.mkdir(parents=True, exist_ok=True)
            split_rng = np.random.default_rng(split_seed)
            # train/validation/test 分别均衡，防止某个 split 的规模影响另一个 split。
            parent_selection_counts = np.zeros(len(features), dtype=np.int64)
            pending: List[Dict[str, np.ndarray]] = []
            shard_index = 0
            for local_index in range(episode_count):
                # seed 只写入分片外的生成元数据，支持逐 episode 重建。
                episode_seed = int(split_rng.integers(0, np.iinfo(np.uint32).max))
                rng = np.random.default_rng(episode_seed)
                episode_arrays, episode_metadata = make_episode(
                    args, calibration, rng, parent_selection_counts
                )
                position_in_shard = len(pending)
                pending.append(episode_arrays)
                metadata = {
                    "episode_id": total_episodes,
                    "source_type": "full_synthetic",
                    "split": split,
                    "shard": f"{split}/shard_{shard_index:05d}.npz",
                    "position_in_shard": position_in_shard,
                    "episode_seed": episode_seed,
                    "market_state_dim": 0,
                    **episode_metadata,
                }
                metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                total_episodes += 1
                is_last = local_index == episode_count - 1
                # 分片可减少小文件数量，同时避免一次在内存积累整个数据集。
                if len(pending) == args.episodes_per_shard or is_last:
                    relative_path = Path(split) / f"shard_{shard_index:05d}.npz"
                    stack_and_save(args.output_dir / relative_path, pending)
                    shard_files.append(str(relative_path))
                    pending = []
                    shard_index += 1
            parent_selection_counts_by_split[split] = parent_selection_counts.tolist()

    os.replace(metadata_temporary, metadata_path)

    # manifest 记录分布匹配方法、数组协议、K=0 约定和所有生成参数。
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset_name": "GKX-calibrated fully synthetic CausalRank SCM episodes",
        "source_type": "full_synthetic",
        "reference_csv": str(args.input_csv.resolve()),
        "calibration_file": "reference_calibration.json",
        "calibration_source": None if calibration_source is None else str(calibration_source),
        "calibration_origin": calibration_origin,
        "feature_names": features,
        "dimensions": {"N": args.num_assets, "T": args.time_steps, "D": len(features)},
        "market_state_dim": 0,
        "contains_C": False,
        "C_handling": "Pass C=None and construct the model with market_state_dim=0.",
        "distribution_matching": {
            "marginal": "monthly cross-sectional ranks mapped to [-1,1]",
            "cross_feature": "sparse SCM edge sampling weighted by real feature correlations",
            "temporal": "feature AR coefficients calibrated from real lag-1 correlations",
            "missingness": "feature rates calibrated from real masks with persistent missing states",
            "asset_coverage": "contiguous listing spells calibrated to real aligned episodes",
        },
        "mask_semantics": {
            "asset_mask": "1 iff the synthetic stock is listed in that period",
            "feature_mask": "1 iff listed and the synthetic characteristic is observed",
            "parent_candidate_mask": "1 iff the feature meets both parent observation thresholds; z/tau losses must be restricted to this set",
            "target_mask": "1 iff synthetic Y is valid; identical to asset_mask in these episodes",
            "time_padding_mask": "PyTorch convention: 1 iff every asset is absent and the time step must be ignored",
        },
        "tau_semantics": "controlled direct q20-to-q80 effect, averaged over asset_mask=1",
        "parent_sampling": {
            "minimum_observations": args.min_parent_observations,
            "minimum_observation_rate": args.min_parent_observation_rate,
            "method": "least-selected eligible features first, with random tie-breaking, balanced independently within each split",
            "selection_counts_by_split": parent_selection_counts_by_split,
        },
        "episode_counts": requested,
        "episodes_per_shard": args.episodes_per_shard,
        "shards": shard_files,
        "training_arrays": {
            "X": "float32 [E,N,T,D]",
            "Y": "float32 [E,N,T]",
            "z": "bool [E,D]",
            "tau_direct": "float32 [E,D]",
            "asset_mask": "bool [E,N,T]",
            "feature_mask": "bool [E,N,T,D]",
            "parent_candidate_mask": "bool [E,D]",
            "target_mask": "bool [E,N,T]",
            "time_padding_mask": "bool [E,T]",
        },
        "generation_metadata": "generation_metadata.jsonl.gz",
        "readme": "README.md",
        "seed": args.seed,
        "generation_arguments": vars(args) | {
            "input_csv": str(args.input_csv),
            "output_dir": str(args.output_dir),
            "cache_dir": str(args.cache_dir),
            "calibration": None if args.calibration is None else str(args.calibration),
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    write_dataset_readme(args.output_dir, manifest)
    print(f"Generated {total_episodes} fully synthetic episodes in {args.output_dir}")
    print(f"Shape per episode: N={args.num_assets}, T={args.time_steps}, D={len(features)}, K=0")
    print("Current GKX DataShare has no C; shards intentionally omit a C array.")


if __name__ == "__main__":
    main()
