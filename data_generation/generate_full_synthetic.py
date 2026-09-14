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
相关的特征对，AR 系数、缺失率和资产覆盖率也由真实训练期校准。为了避免
高持久性与同刻 DAG 传播叠加后让几乎所有 X 都高度共线，生成器会分别缩放
结构边和 AR 系数；缩放量作为命令行参数及 manifest 字段显式保存。生成完整
潜在 X 后，正式训练输入直接使用 ``complete_rank_x``，不再先制造缺失再
补全。真实分布校准得到的持续性缺失过程仍会独立生成，但只以 bit-packed
审计分片保存，不进入训练 DataLoader，也不影响父节点候选资格。

Y 与监督
--------
目标机制与半合成脚本共用同一实现，输出 z 和 q20→q80 受控直接效应
tau_direct。全合成 X 图已知，因此以后可以扩展总效应 tau_total；当前训练
分片只保存论文主任务需要的直接效应，不把祖先总效应与直接父节点混用。
父节点从完整的 94 维统一候选池中均匀无放回抽样；父节点数量在给定区间内逐
episode 随机变化。训练分片保存 ``parent_candidate_mask``。

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# NumPy 负责 SCM、掩码、张量及 NPZ 数据处理。
import numpy as np

# 复用半合成脚本的缓存、秩变换、Y 机制和统一分片协议，避免两套标签定义漂移。
from generate_semi_synthetic import (
    CalibrationAccumulator,
    MIN_QUANTILE_OBSERVATIONS,
    atomic_json,
    atomic_npz,
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


# v4 不改变 Dataset 的训练数组形状，但明确把完整合成面板作为 Full 的训练
# 视图；真实缺失掩码只作审计。这一版本号用于防止与旧 hot-deck Full 数据混淆。
FORMAT_VERSION = 4


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
    parser.add_argument("--min-parents", type=int, default=10)
    parser.add_argument("--max-parents", type=int, default=20)
    parser.add_argument(
        "--test-parent-count",
        type=int,
        default=0,
        help=(
            "测试集固定父节点数；0 表示沿用训练/验证的 min-max 分布。"
            "正式上界测试可设为 20。"
        ),
    )
    parser.add_argument(
        "--parent-sampling-method",
        choices=("uniform", "balanced"),
        default="uniform",
        help=(
            "uniform 在每个 episode 从全部 D 个特征中无放回均匀抽样；"
            "balanced 仅为复现旧数据保留。"
        ),
    )
    parser.add_argument(
        "--min-parent-observations", type=int, default=16,
        help="完整训练面板的最低有效股票月数；不足时拒绝生成。",
    )
    parser.add_argument(
        "--min-parent-observation-rate", type=float, default=0.50,
        help="完整训练面板的最低观测率一致性检查；正式 Full 中应恒为 1。",
    )
    parser.add_argument("--snr-low", type=float, default=0.5)
    parser.add_argument("--snr-high", type=float, default=5.0)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.50)
    # 特征动态 SCM、burn-in 与缺失 Markov 过程的控制参数。
    parser.add_argument("--expected-indegree", type=float, default=2.0)
    parser.add_argument("--max-indegree", type=int, default=5)
    parser.add_argument(
        "--structural-coefficient-scale",
        type=float,
        default=0.65,
        help=(
            "同时刻 DAG 边系数的全局缩放。默认 0.65 用于保留结构相关性，"
            "同时避免完整面板中相关性沿多层 DAG 普遍放大。"
        ),
    )
    parser.add_argument(
        "--ar-coefficient-scale",
        type=float,
        default=0.85,
        help=(
            "真实 lag-1 校准系数的缩放。默认 0.85 保留时间持续性，但降低"
            "高 AR 与同刻结构传播叠加造成的全局共线性。"
        ),
    )
    parser.add_argument(
        "--max-absolute-ar-coefficient",
        type=float,
        default=0.95,
        help="缩放后 AR 系数的绝对值上限，必须小于 1。",
    )
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
    parser.add_argument(
        "--correlation-audit-rows",
        type=int,
        default=4096,
        help=(
            "每个 episode 最多抽取多少个有效股票月计算 X 相关性审计；"
            "0 表示关闭。审计使用独立随机流，不改变 X、Y 或父节点。"
        ),
    )
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
    if not 0 <= args.test_parent_count <= num_features:
        raise ValueError("--test-parent-count must satisfy 0 <= count <= D")
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
    if args.structural_coefficient_scale <= 0.0:
        raise ValueError("--structural-coefficient-scale must be positive")
    if args.ar_coefficient_scale <= 0.0:
        raise ValueError("--ar-coefficient-scale must be positive")
    if not 0.0 < args.max_absolute_ar_coefficient < 1.0:
        raise ValueError("--max-absolute-ar-coefficient must be in (0, 1)")
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
    if args.correlation_audit_rows < 0:
        raise ValueError("--correlation-audit-rows cannot be negative")


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

    # 使用带用途编号的独立随机流。它与后续 split/episode 随机流没有重叠，
    # 因而增加校准窗口数量不会改变正式合成任务的结构和标签抽样。
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 100]))
    accumulator = CalibrationAccumulator(len(features))
    successful = 0
    if args.calibration_windows > starts.size:
        raise ValueError(
            f"requested {args.calibration_windows} unique calibration windows, "
            f"but the GKX training period only provides {starts.size}"
        )
    # 无放回遍历随机排列后的窗口起点，保证成功窗口对应不同的日历区间。
    # 个别窗口可能因股票覆盖不足被跳过，因此遍历全部起点而非只取前 K 个。
    for start_value in rng.permutation(starts):
        if successful >= args.calibration_windows:
            break
        start = int(start_value)
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

    if successful < args.calibration_windows:
        raise RuntimeError(
            f"only {successful} unique calibration windows contained eligible "
            f"stocks; requested {args.calibration_windows}"
        )

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
    structural_coefficient_scale: float,
    ar_coefficient_scale: float,
    max_absolute_ar_coefficient: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """采样稀疏有向无环 SEM，并偏向真实数据中高相关的特征对。

    adjacency[parent, child]=True。先抽拓扑序再只从前驱中抽父节点，因此
    无需事后删环；每条边、AR 系数和节点非线性都会随 episode 保存。

    ``structural_coefficient_scale`` 与 ``ar_coefficient_scale`` 分别控制同刻
    结构传播和时间持续性。二者必须分开控制：只降低 DAG 边仍可能因接近 1
    的 AR 系数跨期积累出全局强相关，只降低 AR 又会损失横截面代理结构。
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
            # 先按入度归一化，再统一缩放。这样高入度节点不会仅因父节点多就
            # 获得更大的结构方差，而 scale 可以直接控制全局传播强度。
            coefficient = (
                sign
                * rng.uniform(0.15, 0.65)
                * structural_coefficient_scale
                / math.sqrt(indegree)
            )
            adjacency[parent, child] = True
            coefficients[parent, child] = coefficient
    # 用真实 lag-1 相关作为 AR 中心并保留 episode 级扰动，然后整体缩放。
    # 缩放保留每列持续性的相对顺序和正负号，不像硬设统一 rho 那样丢失
    # 校准信息；最终再限制绝对值以确保稳定动态。
    calibrated_ar = (
        np.nan_to_num(lag_correlation, nan=0.70)
        + rng.normal(0.0, 0.04, num_features)
    ) * ar_coefficient_scale
    ar_coefficients = np.clip(
        calibrated_ar,
        -max_absolute_ar_coefficient,
        max_absolute_ar_coefficient,
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


def summarize_generated_dependence(
    x: np.ndarray,
    asset_mask: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    """审计最终训练视图中的跨特征相关性与条件数。

    校准矩阵只影响 DAG 边的抽样概率，不能保证多层结构方程生成后的相关性
    仍与校准矩阵同量级。因此审计必须作用于最终的 ``complete_rank_x``，而
    不是只检查输入校准量。最多抽取 ``max_rows`` 个有效股票月，避免在 4096
    episode 数据上为诊断执行不必要的大矩阵乘法。

    此函数只读取 X，并使用独立 RNG；它不会改变结构、目标或父节点随机流。
    """

    if max_rows == 0:
        return None
    valid_rows = np.asarray(x[asset_mask], dtype=np.float64)
    if valid_rows.shape[0] > max_rows:
        selected = rng.choice(valid_rows.shape[0], size=max_rows, replace=False)
        valid_rows = valid_rows[selected]
    if valid_rows.shape[0] < 2:
        raise RuntimeError("not enough valid rows for feature-correlation audit")

    correlation = np.corrcoef(valid_rows, rowvar=False)
    # 完整 rank X 正常情况下每列都有方差；这里仍显式拒绝退化列，防止 NaN
    # 被后续 quantile 静默吞掉并生成表面正常的 manifest。
    if not np.isfinite(correlation).all():
        raise RuntimeError("non-finite generated feature correlation detected")
    upper = np.abs(
        correlation[np.triu_indices(correlation.shape[0], k=1)]
    )
    quantile_levels = (0.50, 0.75, 0.90, 0.95, 0.99)
    quantiles = np.quantile(upper, quantile_levels)
    # 在相关矩阵上加入与 Parent Interaction 默认一致量级的 ridge，再记录
    # 条件数。它不是训练输入，只用于比较不同生成版本的条件识别难度。
    ridge_correlation = correlation + 0.01 * np.eye(correlation.shape[0])
    return {
        "sampled_valid_rows": int(valid_rows.shape[0]),
        "absolute_correlation_quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
        "absolute_correlation_maximum": float(upper.max(initial=0.0)),
        "ridge_correlation_condition_number": float(
            np.linalg.cond(ridge_correlation)
        ),
    }


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
    episode_seed: int,
    parent_selection_counts: np.ndarray,
    min_parents: Optional[int] = None,
    max_parents: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], np.ndarray]:
    """生成一个完整全合成 episode 及其可序列化元数据。

    顺序为 A → 特征 DAG → 完整潜在 X → 截面秩 → M → 已知目标方程。
    输出还保存 SCM 图和参数，使 X 的因果结构能够被复核或用于辅助监督。
    """

    # 结构和目标使用由 episode seed 派生的独立随机流。缺失审计掩码消耗的
    # 随机数不会影响目标随机流，因而修改审计协议不会间接改变父节点集合。
    rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 0]))
    target_rng = np.random.default_rng(
        np.random.SeedSequence([episode_seed, 2])
    )
    correlation_audit_rng = np.random.default_rng(
        np.random.SeedSequence([episode_seed, 3])
    )

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
        args.expected_indegree, args.max_indegree,
        args.structural_coefficient_scale,
        args.ar_coefficient_scale,
        args.max_absolute_ar_coefficient,
        rng,
    )
    latent = generate_latent_x(
        args.num_assets, args.time_steps, args.burn_in,
        adjacency, scm_coefficients, ar_coefficients, order, node_nonlinearity, rng,
    )
    # complete_rank_x 是 Full 数据的正式训练视图。它在每个有效股票月对全部
    # D 个合成特征都有值，因此不会像“遮蔽后补全”那样破坏已知 SCM 关系。
    complete_rank_x = cross_sectional_rank_dense(latent, asset_mask)
    feature_mask = np.broadcast_to(
        asset_mask[:, :, None], complete_rank_x.shape
    ).copy()
    X = np.where(feature_mask, complete_rank_x, 0.0).astype(np.float32, copy=False)
    # 对最终训练视图而非潜变量做审计。独立随机流保证开关审计或改变抽样行数
    # 不会改变目标机制，因此不同设置仍可进行严格的标签对照。
    feature_dependence_audit = summarize_generated_dependence(
        X,
        asset_mask,
        args.correlation_audit_rows,
        correlation_audit_rng,
    )

    # 仍按真实 GKX 观测率生成原始缺失视图，但它只用于检查缺失率和后续
    # missingness ablation。该掩码既不遮蔽 X，也不参与候选/标签生成。
    observed_feature_mask = generate_feature_mask(
        asset_mask, feature_observation_rate, args.missing_persistence, rng
    )

    # 默认使用训练分布的 K 区间；main 可为 test 显式传入固定 K。测试集覆盖
    # 训练区间上界时属于预先声明的条件评估，不会反向改变训练标签分布。
    episode_min_parents = args.min_parents if min_parents is None else min_parents
    episode_max_parents = args.max_parents if max_parents is None else max_parents

    # 与半合成数据共用目标生成器，确保 z/tau 的含义完全一致。
    target_arrays, target_metadata = synthesize_target(
        X, asset_mask, feature_mask, target_rng,
        episode_min_parents, episode_max_parents,
        args.min_parent_observations,
        args.min_parent_observation_rate,
        parent_selection_counts,
        args.snr_low, args.snr_high,
        args.hard_negative_threshold,
        parent_sampling_method=args.parent_sampling_method,
    )
    # Full 的候选定义必须与真实缺失过程完全解耦。只要股票月有效，每个特征
    # 的观测数和观测率都相同，因此这里应严格得到 D 个候选；失败就立即停止
    # 生成，避免悄悄产出带列身份/缺失率标签捷径的数据。
    if not bool(target_arrays["parent_candidate_mask"].all()):
        candidate_count = int(target_arrays["parent_candidate_mask"].sum())
        raise RuntimeError(
            "complete Full panel must make every feature a parent candidate: "
            f"found {candidate_count}/{X.shape[-1]}; check asset coverage and "
            "--min-parent-observations"
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
    valid_value_count = max(
        int(asset_mask.sum()) * observed_feature_mask.shape[-1], 1
    )
    original_observation_counts = observed_feature_mask.sum(
        axis=(0, 1)
    ).astype(np.int64)
    edge_indices = np.argwhere(adjacency)
    metadata = {
        "asset_observation_rate": float(asset_mask.mean()),
        "feature_observation_rate_given_asset": float(
            observed_feature_mask.sum() / valid_value_count
        ),
        "usable_feature_rate_given_asset": float(feature_mask.sum() / valid_value_count),
        "training_feature_source": "complete_rank_x",
        "imputation_method": "none_complete_synthetic_panel",
        "imputed_value_count": 0,
        "imputed_fraction_of_usable_values": 0.0,
        "original_feature_observation_counts": original_observation_counts.tolist(),
        "original_feature_observation_rates": (
            original_observation_counts / max(int(asset_mask.sum()), 1)
        ).tolist(),
        "structure_rng_stream": "SeedSequence([episode_seed, 0])",
        "missingness_audit_rng_stream": "structure RNG stream after X generation",
        "target_rng_stream": "SeedSequence([episode_seed, 2])",
        "feature_edge_count": int(adjacency.sum()),
        "feature_expected_indegree": args.expected_indegree,
        "structural_coefficient_scale": args.structural_coefficient_scale,
        "ar_coefficient_scale": args.ar_coefficient_scale,
        "max_absolute_ar_coefficient": args.max_absolute_ar_coefficient,
        "burn_in": args.burn_in,
        "feature_dependence_audit": feature_dependence_audit,
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
    # 精确原始掩码只用于复现/消融，沿 D 维压缩为 ceil(D/8) 个字节，避免
    # 将额外布尔张量放进每个训练 batch。bitorder 固定后可用 np.unpackbits 恢复。
    observed_feature_mask_packed = np.packbits(
        observed_feature_mask, axis=-1, bitorder="little"
    )
    return arrays, metadata, observed_feature_mask_packed


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
    root_seed = np.random.SeedSequence([args.seed, 200])
    split_seed_sequences = root_seed.spawn(len(requested))
    metadata_path = args.output_dir / "generation_metadata.jsonl.gz"
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    shard_files: List[str] = []
    observed_mask_shard_files: List[str] = []
    total_episodes = 0
    parent_selection_counts_by_split: Dict[str, List[int]] = {}
    parent_count_histogram_by_split: Dict[str, Dict[str, int]] = {}
    dependence_audit_summary_by_split: Dict[str, Dict[str, Any]] = {}

    with gzip.open(metadata_temporary, "wt", encoding="utf-8", newline="\n") as metadata_file:
        for (split, episode_count), split_seed in zip(requested.items(), split_seed_sequences):
            split_directory = args.output_dir / split
            split_directory.mkdir(parents=True, exist_ok=True)
            split_rng = np.random.default_rng(split_seed)
            # train/validation 学习 K=10--20 的可变稀疏度；可选的 test 固定 K
            # 只改变评估条件。两个边界随后写入 manifest，避免分布差异被隐藏。
            if split == "test" and args.test_parent_count > 0:
                split_min_parents = args.test_parent_count
                split_max_parents = args.test_parent_count
            else:
                split_min_parents = args.min_parents
                split_max_parents = args.max_parents
            # 计数始终保存用于审计；uniform 模式不会用历史次数影响父节点抽样。
            parent_selection_counts = np.zeros(len(features), dtype=np.int64)
            parent_count_histogram = np.zeros(split_max_parents + 1, dtype=np.int64)
            split_dependence_audits: List[Mapping[str, Any]] = []
            pending: List[Dict[str, np.ndarray]] = []
            pending_observed_masks: List[np.ndarray] = []
            shard_index = 0
            for local_index in range(episode_count):
                # seed 只写入分片外的生成元数据，支持逐 episode 重建。
                episode_seed = int(split_rng.integers(0, np.iinfo(np.uint32).max))
                episode_arrays, episode_metadata, observed_mask_packed = make_episode(
                    args,
                    calibration,
                    episode_seed,
                    parent_selection_counts,
                    min_parents=split_min_parents,
                    max_parents=split_max_parents,
                )
                position_in_shard = len(pending)
                pending.append(episode_arrays)
                pending_observed_masks.append(observed_mask_packed)
                parent_count_histogram[int(episode_arrays["z"].sum())] += 1
                dependence_audit = episode_metadata["feature_dependence_audit"]
                if dependence_audit is not None:
                    split_dependence_audits.append(dependence_audit)
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
                    observed_relative_path = (
                        Path("audit_observed_feature_mask")
                        / split
                        / f"shard_{shard_index:05d}.npz"
                    )
                    atomic_npz(
                        args.output_dir / observed_relative_path,
                        observed_feature_mask_packed=np.stack(
                            pending_observed_masks
                        ).astype(np.uint8, copy=False),
                    )
                    observed_mask_shard_files.append(str(observed_relative_path))
                    pending = []
                    pending_observed_masks = []
                    shard_index += 1
            parent_selection_counts_by_split[split] = parent_selection_counts.tolist()
            parent_count_histogram_by_split[split] = {
                str(parent_count): int(parent_count_histogram[parent_count])
                for parent_count in range(
                    split_min_parents, split_max_parents + 1
                )
            }
            if split_dependence_audits:
                quantile_names = ("q50", "q75", "q90", "q95", "q99")
                condition_numbers = np.asarray(
                    [
                        audit["ridge_correlation_condition_number"]
                        for audit in split_dependence_audits
                    ],
                    dtype=np.float64,
                )
                dependence_audit_summary_by_split[split] = {
                    "audited_episodes": len(split_dependence_audits),
                    "rows_per_episode_at_most": args.correlation_audit_rows,
                    "mean_absolute_correlation_quantiles": {
                        name: float(
                            np.mean(
                                [
                                    audit["absolute_correlation_quantiles"][name]
                                    for audit in split_dependence_audits
                                ]
                            )
                        )
                        for name in quantile_names
                    },
                    "ridge_correlation_condition_number": {
                        "median": float(np.median(condition_numbers)),
                        "p90": float(np.quantile(condition_numbers, 0.90)),
                        "maximum": float(condition_numbers.max()),
                    },
                }

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
            "cross_feature": (
                "sparse SCM edge sampling weighted by real feature correlations; "
                "edge coefficients are globally scaled to prevent pervasive "
                "multi-hop collinearity"
            ),
            "temporal": (
                "feature AR coefficients calibrated from real lag-1 correlations, "
                "then scaled and stability-clipped before simulation"
            ),
            "missingness": (
                "raw feature rates are calibrated from real masks with persistent "
                "missing states, but this mask is audit-only and never modifies the "
                "formal complete_rank_x training input or parent eligibility"
            ),
            "asset_coverage": "contiguous listing spells calibrated to real aligned episodes",
        },
        "feature_dependence_control": {
            "structural_coefficient_scale": args.structural_coefficient_scale,
            "ar_coefficient_scale": args.ar_coefficient_scale,
            "max_absolute_ar_coefficient": args.max_absolute_ar_coefficient,
            "audit_semantics": (
                "absolute Pearson correlations are computed on sampled valid rows "
                "from final complete_rank_x; the ridge condition number uses "
                "corr(X)+0.01*I"
            ),
            "summary_by_split": dependence_audit_summary_by_split,
        },
        "mask_semantics": {
            "asset_mask": "1 iff the synthetic stock is listed in that period",
            "feature_mask": (
                "broadcast(asset_mask): every one of the D synthetic features is "
                "available at every listed stock-month"
            ),
            "parent_candidate_mask": (
                "all True [D] by construction: every synthetic feature has equal "
                "eligibility; z/tau losses are defined over all D features"
            ),
            "target_mask": "1 iff synthetic Y is valid; identical to asset_mask in these episodes",
            "time_padding_mask": "PyTorch convention: 1 iff every asset is absent and the time step must be ignored",
        },
        "training_feature_view": {
            "source": "complete_rank_x",
            "imputation": "none",
            "all_features_available_when_asset_is_listed": True,
            "raw_missingness_used_for_training": False,
        },
        "tau_semantics": "controlled direct q20-to-q80 effect, averaged over asset_mask=1",
        "parent_sampling": {
            "minimum_observations": args.min_parent_observations,
            "minimum_observation_rate": args.min_parent_observation_rate,
            "minimum_parent_count": args.min_parents,
            "maximum_parent_count": args.max_parents,
            "test_parent_count": (
                args.test_parent_count if args.test_parent_count > 0 else None
            ),
            "count_distribution_by_split": {
                "train": "discrete uniform on [minimum_parent_count, maximum_parent_count]",
                "validation": "discrete uniform on [minimum_parent_count, maximum_parent_count]",
                "test": (
                    f"fixed at {args.test_parent_count}"
                    if args.test_parent_count > 0
                    else "discrete uniform on [minimum_parent_count, maximum_parent_count]"
                ),
            },
            "method": args.parent_sampling_method,
            "method_description": (
                "uniform sampling without replacement from all D features in every "
                "episode; raw missingness never changes eligibility"
                if args.parent_sampling_method == "uniform"
                else "least-selected eligible features first, with random "
                "tie-breaking, balanced independently within each split"
            ),
            "selection_counts_by_split": parent_selection_counts_by_split,
            "actual_count_histogram_by_split": parent_count_histogram_by_split,
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
        "audit_arrays": {
            "observed_feature_mask": {
                "semantics": (
                    "audit-only synthetic observation mask calibrated from raw GKX; "
                    "it never masks complete_rank_x or changes parent eligibility"
                ),
                "encoding": "np.packbits(axis=-1, bitorder='little')",
                "stored_key": "observed_feature_mask_packed",
                "packed_shape": "uint8 [E,N,T,ceil(D/8)]",
                "restore": "np.unpackbits(packed, axis=-1, count=D, bitorder='little').astype(bool)",
                "shards": observed_mask_shard_files,
                "loaded_by_training_dataset": False,
            }
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
