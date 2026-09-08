#!/usr/bin/env python3
"""从 GKX DataShare 生成日历月对齐的半合成 CausalRank episode。

数据含义
--------
GKX 原始文件包含 ``permno``、``DATE``、94 个经过滞后处理的公司特征和
``sic2``，不包含收益 Y，也不包含全市场共享的宏观状态 C。本程序保留真实
X、股票覆盖情况和特征缺失模式，只合成目标方程 Y 及其监督标签。

半合成数据中的困难负样本
--------------------------
程序只使用一种安全方式：在真实 X 中寻找与父节点高度相关、但没有进入
目标方程的非父节点，并在 ``hard_negative_mask`` 中标记。程序不会向缺失
位置注入父节点信号，也不会宣称已知真实 X 之间的因果图。因此这里得到的
是“自然相关困难负样本”，不能仅凭观测数据断言它一定是真实混杂变量。

每个压缩 NPZ 分片的核心张量
-----------------------------
``X`` [E,N,T,D]、``Y`` [E,N,T]、``z`` [E,D]、
``tau_direct`` [E,D]、``asset_mask`` [E,N,T]、
``feature_mask`` [E,N,T,D]、``parent_candidate_mask`` [E,D]、
``target_mask`` [E,N,T] 和
``time_padding_mask`` [E,T]。E 是分片内的 episode 数。

由于当前 GKX 没有 C，manifest 固定记录 ``market_state_dim=0``，分片不保存
伪造的空 C；训练时应设置 ``market_state_dim=0`` 并传入 ``C=None``。

保存格式
--------
训练张量保存为分片压缩 NPZ，episode 级生成机制信息保存为压缩 JSONL，
校准摘要保存为 JSON；数据定义、特征顺序、划分范围和随机种子保存为
manifest.json，并自动生成 README.md。训练分片只含模型训练所需的 9 个数组。
"""

from __future__ import annotations  # 推迟类型标注求值，兼容项目的 Python 3.9。

# argparse 负责命令行参数；hashlib 用于给缓存配置生成稳定指纹。
import argparse
import gzip
import hashlib
# json 保存 manifest/episode 元数据；math 提供窗口和数值计算。
import json
import math
# os.replace 用于原子替换临时文件，避免中断留下半成品。
import os
# 相邻 episode 会重复使用月份，LRU 缓存可减少磁盘解压。
from functools import lru_cache
# Path 使默认路径不依赖运行命令的当前目录。
from pathlib import Path
# 类型标注用于明确元数据和数组容器的结构。
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# NumPy 负责张量计算/NPZ；pandas 负责流式 CSV 和并列秩。
import numpy as np
import pandas as pd


# 数据集输出协议新增 parent_candidate_mask，因此升级到版本 2。
FORMAT_VERSION = 2
# 月度源数据缓存格式没有改变，单独保留版本 1，避免无意义地重建 3.8GB 缓存。
CACHE_FORMAT_VERSION = 1
# GKX 的股票永久标识符和月末日期列。
ID_COLUMN = "permno"
DATE_COLUMN = "DATE"
# 标识符、日期和行业码不属于 94 个连续候选因子。
NON_FEATURE_COLUMNS = {ID_COLUMN, DATE_COLUMN, "sic2"}
# 目标方程中单个父节点可采用的随机基函数。
TRANSFORM_NAMES = ("linear", "tanh", "signed_square", "sine", "threshold")
# q20/q80 至少基于这些真实观测计算，父候选阈值不得低于它。
MIN_QUANTILE_OBSERVATIONS = 16
# 这是训练分片允许出现的完整键集合；生成参数必须放到分片外部。
TRAINING_ARRAY_NAMES = (
    "X", "Y", "z", "tau_direct", "asset_mask", "feature_mask",
    "parent_candidate_mask", "target_mask", "time_padding_mask",
)


def find_project_root() -> Path:
    """从脚本当前位置向上查找 CausalRank 根目录。

    生成脚本可能放在 ``CausalRank/data_generation``，也可能放在旧位置
    ``CausalRank/data/data_generation``。通过项目中的 ``article/proposal``
    标记定位根目录，避免再依赖固定的 ``parents[n]`` 层级。
    """

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "article" / "proposal").is_file():
            return candidate
    raise RuntimeError(
        "cannot locate the CausalRank project root from the generator script"
    )


def parse_args() -> argparse.Namespace:
    """定义数据规模、日期划分、目标机制和保存位置。"""

    # 自动定位项目根目录，因此移动 data_generation 文件夹后仍能找到真实数据。
    project_root = find_project_root()
    script_directory = Path(__file__).resolve().parent
    # 默认使用项目中已经下载并解压的 GKX 数据。
    default_data = project_root / "data" / "GKX Characteristics Data"
    parser = argparse.ArgumentParser(
        description="从 GKX DataShare 生成日历月对齐的半合成 episode。"
    )
    # 输入、输出和可复用逐月缓存的位置。
    parser.add_argument("--input-csv", type=Path, default=default_data / "datashare.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=script_directory / "semi_synthetic",
        help="输出目录；相对路径按运行命令时的当前目录解析。",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=default_data / "monthly_npz_cache",
        help="可复用、带版本指纹的逐月 NPZ 缓存根目录。",
    )
    # 每个 episode 的 N/T，以及每个 split 和分片包含的样本数。
    parser.add_argument("--num-assets", type=int, default=256)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--train-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--episodes-per-shard", type=int, default=8)
    # 先按月份比例切分原始数据，再在各 split 内抽连续窗口。
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-asset-coverage", type=float, default=0.80)
    # 稀疏目标方程、信噪比和自然相关困难负样本阈值。
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
    # 随机性、CSV 峰值内存和仅供烟雾测试使用的截断开关。
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument(
        "--max-source-rows", type=int, default=0,
        help="最多读取的源数据行数；0 表示全部，仅建议在烟雾测试时截断。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在读取 3.8GB 原始文件之前尽早拒绝错误参数。"""

    if not args.input_csv.is_file():
        raise FileNotFoundError(f"GKX CSV does not exist: {args.input_csv}")
    for name in ("num_assets", "time_steps", "episodes_per_shard", "chunksize"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("train_episodes", "validation_episodes", "test_episodes"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1]")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if args.train_fraction + args.validation_fraction > 1.0:
        raise ValueError("train and validation fractions cannot sum to more than 1")
    if not 0.0 < args.min_asset_coverage <= 1.0:
        raise ValueError("--min-asset-coverage must be in (0, 1]")
    if not 1 <= args.min_parents <= args.max_parents <= 94:
        raise ValueError("parent bounds must satisfy 1 <= min <= max <= 94")
    if not MIN_QUANTILE_OBSERVATIONS <= args.min_parent_observations <= args.num_assets * args.time_steps:
        raise ValueError(
            f"--min-parent-observations must be in [{MIN_QUANTILE_OBSERVATIONS}, "
            "num-assets * time-steps]"
        )
    if not 0.0 < args.min_parent_observation_rate <= 1.0:
        raise ValueError("--min-parent-observation-rate must be in (0, 1]")
    if not 0.0 < args.snr_low <= args.snr_high:
        raise ValueError("SNR bounds must satisfy 0 < low <= high")
    if args.max_source_rows < 0:
        raise ValueError("--max-source-rows cannot be negative")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """先写临时 JSON，再原子替换目标，避免中断产生半个 manifest。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """原子写入 UTF-8 文本，供生成目录内的 README 使用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, path)


def write_dataset_readme(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    """根据 manifest 在数据目录中生成自包含的中文数据说明。"""

    dimensions = manifest["dimensions"]
    counts = manifest["episode_counts"]
    parent_sampling = manifest["parent_sampling"]
    arguments = json.dumps(
        manifest["generation_arguments"], ensure_ascii=False, indent=2, sort_keys=True
    )
    calibration_name = manifest.get("calibration_file", "无")
    if manifest.get("calibration_origin") == "external":
        calibration_description = (
            f"`{calibration_name}`：本次实际使用的外部校准参数快照；"
            "原始路径记录在 `manifest.json`"
        )
    else:
        calibration_description = f"`{calibration_name}`"
    content = f"""# {manifest['dataset_name']}

## 数据规模与划分

- 单个 episode：`N={dimensions['N']}`，`T={dimensions['T']}`，`D={dimensions['D']}`，`K=0`。
- episode 数：训练 {counts['train']}，验证 {counts['validation']}，测试 {counts['test']}。
- 每个 shard 最多含 {manifest['episodes_per_shard']} 个 episode，因此首维 `E` 可能小于该值。
- 当前数据不含市场状态 `C`；模型应设置 `market_state_dim=0` 并传入 `C=None`。

## 训练分片

`train/`、`validation/`、`test/` 下的 `.npz` 只保存以下训练数组：

| 数组 | dtype | shape | 训练角色 | 含义 |
|---|---|---|---|---|
| `X` | float32 | `[E,N,T,D]` | 模型输入 | 股票特征；缺失和 padding 位置填 0 |
| `Y` | float32 | `[E,N,T]` | 模型输入 | 合成目标；无效位置填 0 |
| `z` | bool | `[E,D]` | 分类监督 | 是否为 `Y` 的直接父特征 |
| `tau_direct` | float32 | `[E,D]` | 排序监督 | q20→q80 的受控直接效应，非父特征为 0 |
| `asset_mask` | bool | `[E,N,T]` | 有效性掩码 | 股票—月份是否存在 |
| `feature_mask` | bool | `[E,N,T,D]` | 有效性掩码 | 特征值是否真实可用 |
| `parent_candidate_mask` | bool | `[E,D]` | 监督候选掩码 | 是否达到父节点最低观测数和观测率 |
| `target_mask` | bool | `[E,N,T]` | 损失掩码 | `Y` 是否参与损失；当前等于 `asset_mask` |
| `time_padding_mask` | bool | `[E,T]` | 注意力掩码 | PyTorch 语义；True 表示整期应忽略 |

标准训练使用全部 9 个数组；其中 `z` 和 `tau_direct` 是监督标签，五种 mask
用于限定候选特征或屏蔽缺失和 padding。分类、排序与效应损失必须限制在
`parent_candidate_mask=True` 的特征内。当前 `target_mask` 可由 `asset_mask` 复制得到，
`time_padding_mask` 可由 `~asset_mask.any(axis=1)` 得到；仍显式保存二者，避免
训练时重复实现掩码语义，并允许以后为真实缺失 `Y` 扩展 `target_mask`。
`q_low`、`q_high` 只定义 tau 的干预端点，不是模型输入或监督标签。

## 父节点候选与均衡抽样

- 特征至少有 {parent_sampling['minimum_observations']} 个有效观测，并且相对有效
  股票月的观测率至少为 {parent_sampling['minimum_observation_rate']:.0%}，才会令
  `parent_candidate_mask=True` 并进入父节点候选集合。
- 每个 train/validation/test split 分别维护94个特征的累计父节点入选次数；
  每个 episode 优先从当前入选次数最少的合格特征中选择，并随机打破并列。
- “均匀”仅针对当期合格候选；不满足完整率的特征不会为了凑均匀而成为父节点。
- 各 split 最终的逐特征入选次数记录在 `manifest.json` 的
  `parent_sampling.selection_counts_by_split`。

## 非训练文件

- `generation_metadata.jsonl.gz`：每行对应一个 episode，包含定位字段、seed、
  q_low/q_high、目标机制参数、困难负样本索引，以及该生成器特有的复现信息。
- `manifest.json`：全局结构、特征顺序、分片清单、mask/tau 语义和完整运行参数。
- {calibration_description}；仅供生成/校准全合成数据使用，不由训练加载器读取。
- `README.md`：本说明。

## 复现定位

元数据中的 `shard` 与 `position_in_shard` 唯一定位训练分片中的一个 episode。
`hard_negative_indices` 是特征下标列表；完整特征名及顺序见 `manifest.json`。

## 本次生成参数

```json
{arguments}
```
"""
    atomic_text(output_dir / "README.md", content)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """原子写入压缩 NPZ；调用方负责保证同名数组的 shape 和 dtype。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def feature_columns(input_csv: Path) -> List[str]:
    """从表头提取并校验顺序固定的 94 个 GKX 候选特征。"""

    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = {ID_COLUMN, DATE_COLUMN}.difference(columns)
    if missing:
        raise ValueError(f"required columns are absent: {sorted(missing)}")
    features = [column for column in columns if column not in NON_FEATURE_COLUMNS]
    if len(features) != 94:
        raise ValueError(f"expected 94 GKX characteristics, found {len(features)}")
    return features


def cache_fingerprint(input_csv: Path, features: Sequence[str], max_rows: int) -> str:
    """根据源文件及读取配置计算缓存版本，避免错误复用旧缓存。"""

    stat = input_csv.stat()
    descriptor = {
        "path": str(input_csv.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "features": list(features),
        "max_rows": max_rows,
        "format_version": CACHE_FORMAT_VERSION,
    }
    encoded = json.dumps(descriptor, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_month_cache(
    input_csv: Path,
    cache_root: Path,
    features: Sequence[str],
    chunksize: int,
    max_rows: int,
) -> Tuple[Path, Dict[str, Any]]:
    """把按日期排序的长表一次性转换为可复用的逐月 NPZ。

    CSV 以 chunk 读取，不会一次把整个 3.8GB 文件载入内存。同一个月可能
    横跨两个 chunk，因此先累计该月片段，直到读到下一月份才落盘。
    """

    # 同一输入和配置映射到同一子目录；源文件变化后自动换新目录。
    fingerprint = cache_fingerprint(input_csv, features, max_rows)
    cache_path = cache_root / fingerprint
    manifest_path = cache_path / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("complete") and manifest.get("feature_names") == list(features):
            return cache_path, manifest

    cache_path.mkdir(parents=True, exist_ok=True)
    usecols = [ID_COLUMN, DATE_COLUMN, *features]
    read_kwargs: Dict[str, Any] = {
        "usecols": usecols,
        "chunksize": chunksize,
        "dtype": {ID_COLUMN: np.int64, DATE_COLUMN: np.int64, **{f: np.float32 for f in features}},
    }
    if max_rows:
        read_kwargs["nrows"] = max_rows

    current_date: Optional[int] = None
    id_parts: List[np.ndarray] = []
    value_parts: List[np.ndarray] = []
    dates: List[int] = []
    total_rows = 0

    def flush_month() -> None:
        """合并当前月片段，按 permno 排序、检查重复键并原子落盘。"""

        nonlocal id_parts, value_parts, current_date
        if current_date is None:
            return
        ids = np.concatenate(id_parts)
        values = np.concatenate(value_parts, axis=0)
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        values = values[order]
        if ids.size > 1 and np.any(ids[1:] == ids[:-1]):
            raise ValueError(f"duplicate permno within DATE={current_date}")
        atomic_npz(
            cache_path / f"month_{current_date}.npz",
            stock_ids=ids.astype(np.int64, copy=False),
            X_obs=values.astype(np.float32, copy=False),
        )
        dates.append(current_date)
        id_parts = []
        value_parts = []

    # datashare.csv 已按 DATE、permno 排序，因此可以流式检测月份边界。
    for chunk in pd.read_csv(input_csv, **read_kwargs):
        if chunk.empty:
            continue
        chunk_dates = chunk[DATE_COLUMN].to_numpy(np.int64, copy=False)
        if np.any(chunk_dates[1:] < chunk_dates[:-1]):
            raise ValueError("input CSV must be sorted by DATE")
        total_rows += len(chunk)
        for date_value, group in chunk.groupby(DATE_COLUMN, sort=False):
            date_int = int(date_value)
            if current_date is not None and date_int < current_date:
                raise ValueError("input CSV must be sorted by DATE")
            if current_date is not None and date_int != current_date:
                flush_month()
            if current_date != date_int:
                current_date = date_int
            id_parts.append(group[ID_COLUMN].to_numpy(np.int64, copy=True))
            value_parts.append(group[list(features)].to_numpy(dtype=np.float32, copy=True))
    flush_month()

    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "complete": True,
        "source_csv": str(input_csv.resolve()),
        "source_size_bytes": input_csv.stat().st_size,
        "max_source_rows": max_rows,
        "rows_cached": total_rows,
        "feature_names": list(features),
        "dates": dates,
        "num_dates": len(dates),
    }
    atomic_json(manifest_path, manifest)
    return cache_path, manifest


def month_ordinals(dates: np.ndarray) -> np.ndarray:
    """把 YYYYMMDD 月末日期转换为便于检查连续性的月份整数。"""

    parsed = pd.to_datetime(dates.astype(str), format="%Y%m%d")
    return pd.PeriodIndex(parsed, freq="M").asi8


def split_calendar(
    dates: np.ndarray, train_fraction: float, validation_fraction: float
) -> Dict[str, np.ndarray]:
    """在生成窗口前按日期顺序切分 train、validation 和 test。"""

    num_dates = len(dates)
    train_end = int(math.floor(num_dates * train_fraction))
    validation_end = train_end + int(math.floor(num_dates * validation_fraction))
    return {
        "train": dates[:train_end],
        "validation": dates[train_end:validation_end],
        "test": dates[validation_end:],
    }


def valid_window_starts(dates: np.ndarray, time_steps: int) -> np.ndarray:
    """返回没有缺月且完整包含 T 个连续自然月的窗口起点。"""

    if len(dates) < time_steps:
        return np.empty(0, dtype=np.int64)
    ordinal = month_ordinals(dates)
    starts = [
        index
        for index in range(len(dates) - time_steps + 1)
        if ordinal[index + time_steps - 1] - ordinal[index] == time_steps - 1
        and np.all(np.diff(ordinal[index : index + time_steps]) == 1)
    ]
    return np.asarray(starts, dtype=np.int64)


def rank_transform(X_obs: np.ndarray, asset_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """逐月截面秩映射至 [-1,1]，缺失值填 0，并返回 feature_mask。

    0 对应秩空间的截面中位数。填 0 只为向模型提供有限数值，原始缺失信息
    始终保存在 feature_mask 中。
    """

    # M=1 要同时满足股票月存在且该特征不是 NaN/Inf。
    feature_mask = np.isfinite(X_obs) & asset_mask[:, :, None]
    transformed = np.zeros(X_obs.shape, dtype=np.float32)
    _, num_times, num_features = X_obs.shape
    for time_index in range(num_times):
        for feature_index in range(num_features):
            valid = feature_mask[:, time_index, feature_index]
            count = int(valid.sum())
            if count <= 1:
                continue
            values = X_obs[valid, time_index, feature_index]
            ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float32)
            transformed[valid, time_index, feature_index] = (
                2.0 * (ranks - 1.0) / float(count - 1) - 1.0
            )
    return transformed, feature_mask


def transformed_value(values: np.ndarray, code: int, threshold: float) -> np.ndarray:
    """执行父节点在目标结构方程中的线性或随机非线性基函数。"""

    if code == 0:
        return values
    if code == 1:
        return np.tanh(2.0 * values)
    if code == 2:
        return np.sign(values) * values * values
    if code == 3:
        return np.sin(np.pi * values)
    if code == 4:
        return (values > threshold).astype(np.float32) - 0.5
    raise ValueError(f"unknown transform code: {code}")


def evaluate_target(
    X: np.ndarray,
    parents: np.ndarray,
    coefficients: np.ndarray,
    transform_codes: np.ndarray,
    thresholds: np.ndarray,
    interaction_pairs: Sequence[Tuple[int, int]],
    interaction_coefficients: np.ndarray,
    noise: np.ndarray,
    noise_scale: float,
    heteroskedastic_parent: int,
    override: Optional[Tuple[int, float]] = None,
) -> np.ndarray:
    """计算事实或单变量反事实目标。

    override=(j,q) 只替换目标方程读取的 X_j；其他 X、噪声和机制参数保持
    不变，因此事实与反事实之差对应受控直接效应。
    """
    # 使用 float64 累加结构项，最终保存前再转换为 float32。
    result = np.zeros(X.shape[:2], dtype=np.float64)

    def value_for(feature_index: int) -> np.ndarray:
        """读取事实特征，或在当前干预变量上返回常数反事实值。"""

        if override is not None and feature_index == override[0]:
            return np.full(X.shape[:2], override[1], dtype=np.float32)
        return X[:, :, feature_index]

    for parent in parents:
        values = value_for(int(parent))
        basis = transformed_value(values, int(transform_codes[parent]), float(thresholds[parent]))
        result += float(coefficients[parent]) * basis
    for pair, coefficient in zip(interaction_pairs, interaction_coefficients):
        left, right = pair
        result += float(coefficient) * value_for(left) * value_for(right)

    # 异方差机制让第一个父节点同时调节噪声尺度；反事实会同步替换该驱动。
    local_noise_scale: Union[np.ndarray, float] = noise_scale
    if heteroskedastic_parent >= 0:
        driver = value_for(heteroskedastic_parent)
        local_noise_scale = noise_scale * (0.5 + 0.75 * np.abs(driver))
    return result + local_noise_scale * noise


def hard_negative_mask(
    X: np.ndarray,
    asset_mask: np.ndarray,
    parents: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """标记真实 X 中自然存在的高相关非父节点，不修改任何 X 数值。

    这是半合成脚本唯一采用的困难负样本方式：不在缺失位置注入父节点
    信号，也不额外合成未观测共同原因。由于真实 X 图未知，该标记表示强
    相关非父节点，而不等同于已被证明的真实混杂变量。
    """
    num_features = X.shape[-1]
    answer = np.zeros(num_features, dtype=bool)
    # 只比较真实存在的股票月；缺失特征已经按约定以 0 插补。
    rows = X[asset_mask]
    if rows.shape[0] < 3:
        return answer
    standard_deviation = rows.std(axis=0)
    usable = standard_deviation > 1e-8
    normalized = np.zeros_like(rows, dtype=np.float64)
    normalized[:, usable] = (
        rows[:, usable] - rows[:, usable].mean(axis=0)
    ) / standard_deviation[usable]
    correlation = normalized.T @ normalized / max(rows.shape[0] - 1, 1)
    parent_set = set(int(value) for value in parents)
    nonparents = np.asarray([j for j in range(num_features) if j not in parent_set])
    if nonparents.size == 0:
        return answer
    for parent in parents:
        scores = np.abs(correlation[int(parent), nonparents])
        winner = int(nonparents[int(np.argmax(scores))])
        if float(np.max(scores)) >= threshold:
            answer[winner] = True
    return answer


def synthesize_target(
    X: np.ndarray,
    asset_mask: np.ndarray,
    feature_mask: np.ndarray,
    rng: np.random.Generator,
    min_parents: int,
    max_parents: int,
    min_parent_observations: int,
    min_parent_observation_rate: float,
    parent_selection_counts: np.ndarray,
    snr_low: float,
    snr_high: float,
    negative_threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """生成 Y、直接父节点标签 z 和 q20→q80 直接效应 tau_direct。

    父节点只从达到最低有效观测数和观测率的特征中抽取。为避免完整特征
    长期垄断父节点标签，优先选择当前 split 中累计入选次数较少的候选特征，
    并用 episode RNG 随机打破并列。目标机制覆盖加性、交互和异方差三类。
    """
    num_features = X.shape[-1]
    if parent_selection_counts.shape != (num_features,):
        raise ValueError("parent_selection_counts must have shape [D]")

    # 相对完整率以真实存在的股票月为分母，不把股票 padding 计作特征缺失。
    valid_asset_count = int(asset_mask.sum())
    observation_counts = feature_mask.sum(axis=(0, 1)).astype(np.int64)
    observation_rates = observation_counts / max(valid_asset_count, 1)
    parent_candidate_mask = (
        (observation_counts >= min_parent_observations)
        & (observation_rates >= min_parent_observation_rate)
    )
    candidates = np.flatnonzero(parent_candidate_mask)
    if candidates.size < min_parents:
        raise RuntimeError(
            "not enough relatively complete parent candidates: "
            f"found {candidates.size}, need at least {min_parents}; "
            "lower --min-parent-observations/--min-parent-observation-rate "
            "or choose another episode configuration"
        )

    parent_count = int(
        rng.integers(min_parents, min(max_parents, candidates.size) + 1)
    )
    # 先按累计入选次数升序，再用随机数打破相同次数的并列。
    random_tie_breaker = rng.random(candidates.size)
    balanced_order = np.lexsort(
        (random_tie_breaker, parent_selection_counts[candidates])
    )
    parents = np.sort(candidates[balanced_order[:parent_count]]).astype(np.int64)
    parent_selection_counts[parents] += 1
    z = np.zeros(num_features, dtype=bool)
    z[parents] = True
    coefficients = np.zeros(num_features, dtype=np.float32)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=parent_count)
    coefficients[parents] = signs * rng.uniform(0.4, 1.6, size=parent_count)
    transform_codes = np.full(num_features, -1, dtype=np.int8)
    transform_codes[parents] = rng.integers(0, len(TRANSFORM_NAMES), size=parent_count)
    thresholds = np.zeros(num_features, dtype=np.float32)
    thresholds[parents] = rng.uniform(-0.4, 0.4, size=parent_count)

    # 部分 episode 在父节点之间加入二阶交互，用于覆盖非加性目标机制。
    interaction_pairs: List[Tuple[int, int]] = []
    if parent_count >= 2 and rng.random() < 0.60:
        shuffled = rng.permutation(parents)
        interaction_count = min(parent_count // 2, int(rng.integers(1, 3)))
        interaction_pairs = [
            (int(shuffled[2 * index]), int(shuffled[2 * index + 1]))
            for index in range(interaction_count)
        ]
    interaction_coefficients = rng.uniform(
        -0.8, 0.8, size=len(interaction_pairs)
    ).astype(np.float32)

    mechanism_type = str(rng.choice(np.asarray(["additive", "interaction", "heteroskedastic"])))
    if mechanism_type == "additive":
        interaction_pairs = []
        interaction_coefficients = np.empty(0, dtype=np.float32)
    heteroskedastic_parent = int(parents[0]) if mechanism_type == "heteroskedastic" else -1
    # 先计算无噪声信号方差，再按随机 SNR 反推出外生噪声尺度。
    zero_noise = np.zeros(X.shape[:2], dtype=np.float32)
    signal = evaluate_target(
        X, parents, coefficients, transform_codes, thresholds,
        interaction_pairs, interaction_coefficients, zero_noise, 0.0,
        heteroskedastic_parent,
    )
    valid_signal = signal[asset_mask]
    signal_scale = float(valid_signal.std()) if valid_signal.size else 0.0
    snr = float(np.exp(rng.uniform(np.log(snr_low), np.log(snr_high))))
    noise_scale = max(signal_scale, 0.25) / math.sqrt(snr)
    noise = rng.standard_normal(X.shape[:2]).astype(np.float32)
    y_raw = evaluate_target(
        X, parents, coefficients, transform_codes, thresholds,
        interaction_pairs, interaction_coefficients, noise, noise_scale,
        heteroskedastic_parent,
    )
    # 只用 A=1 的位置拟合 Y 标准化参数；无效位置最终固定为 0。
    valid_y = y_raw[asset_mask]
    y_mean = float(valid_y.mean())
    y_std = max(float(valid_y.std()), 1e-6)
    Y = np.zeros(X.shape[:2], dtype=np.float32)
    Y[asset_mask] = ((valid_y - y_mean) / y_std).astype(np.float32)

    # 干预值优先使用 episode 内观测分位数；样本不足时用秩空间默认值。
    q_low = np.full(num_features, -0.6, dtype=np.float32)
    q_high = np.full(num_features, 0.6, dtype=np.float32)
    for feature_index in range(num_features):
        observed = feature_mask[:, :, feature_index]
        values = X[:, :, feature_index][observed]
        if values.size >= MIN_QUANTILE_OBSERVATIONS:
            q_low[feature_index], q_high[feature_index] = np.quantile(values, [0.2, 0.8])

    # 非父节点保持严格为 0；父节点事实/反事实复用同一 noise。
    tau = np.zeros(num_features, dtype=np.float32)
    for parent in parents:
        low = evaluate_target(
            X, parents, coefficients, transform_codes, thresholds,
            interaction_pairs, interaction_coefficients, noise, noise_scale,
            heteroskedastic_parent, (int(parent), float(q_low[parent])),
        )
        high = evaluate_target(
            X, parents, coefficients, transform_codes, thresholds,
            interaction_pairs, interaction_coefficients, noise, noise_scale,
            heteroskedastic_parent, (int(parent), float(q_high[parent])),
        )
        tau[parent] = float(np.mean(np.abs((high[asset_mask] - low[asset_mask]) / y_std)))

    # 返回值暂含机制参数；main 会把训练必需数组与生成元数据分开保存。
    arrays = {
        "Y": Y,
        "z": z,
        "tau_direct": tau,
        "parent_candidate_mask": parent_candidate_mask,
        "q_low": q_low,
        "q_high": q_high,
        "parent_coefficients": coefficients,
        "transform_codes": transform_codes,
        "transform_thresholds": thresholds,
        "hard_negative_mask": hard_negative_mask(
            X, asset_mask, parents, negative_threshold
        ),
    }
    metadata = {
        "parent_indices": parents.tolist(),
        "parent_candidate_count": int(parent_candidate_mask.sum()),
        "parent_observation_counts": observation_counts[parents].tolist(),
        "parent_observation_rates": observation_rates[parents].tolist(),
        "parent_selection_counts_after": parent_selection_counts[parents].tolist(),
        "parent_transforms": {
            str(int(parent)): TRANSFORM_NAMES[int(transform_codes[parent])]
            for parent in parents
        },
        "interaction_pairs": [list(pair) for pair in interaction_pairs],
        "interaction_coefficients": interaction_coefficients.tolist(),
        "mechanism_type": mechanism_type,
        "heteroskedastic_parent": heteroskedastic_parent,
        "snr": snr,
        "noise_scale": noise_scale,
        "y_raw_mean": y_mean,
        "y_raw_std": y_std,
    }
    return arrays, metadata


class CalibrationAccumulator:
    """累计真实面板的覆盖率、缺失率、时序相关和截面相关统计量。

    统计量最终写入 calibration.json，供全合成生成器校准其 X 分布。这里
    使用累计矩而不是保存全部样本，避免校准过程额外占用大量内存。
    """

    def __init__(self, num_features: int) -> None:
        """为标量、逐特征统计量和 D×D 二阶矩分配累计器。"""

        self.num_features = num_features
        self.asset_observed = 0.0
        self.asset_total = 0.0
        self.feature_observed = np.zeros(num_features, dtype=np.float64)
        self.feature_possible = 0.0
        self.asset_stay_numerator = 0.0
        self.asset_stay_denominator = 0.0
        self.asset_enter_numerator = 0.0
        self.asset_enter_denominator = 0.0
        self.lag_count = np.zeros(num_features, dtype=np.float64)
        self.lag_sum_x = np.zeros(num_features, dtype=np.float64)
        self.lag_sum_y = np.zeros(num_features, dtype=np.float64)
        self.lag_sum_xx = np.zeros(num_features, dtype=np.float64)
        self.lag_sum_yy = np.zeros(num_features, dtype=np.float64)
        self.lag_sum_xy = np.zeros(num_features, dtype=np.float64)
        self.corr_count = 0
        self.corr_sum = np.zeros(num_features, dtype=np.float64)
        self.corr_outer = np.zeros((num_features, num_features), dtype=np.float64)

    def update(
        self,
        X: np.ndarray,
        asset_mask: np.ndarray,
        feature_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """使用一个对齐的 X/A/M episode 更新全部累计统计量。"""

        self.asset_observed += float(asset_mask.sum())
        self.asset_total += float(asset_mask.size)
        self.feature_observed += feature_mask.sum(axis=(0, 1))
        self.feature_possible += float(asset_mask.sum())
        previous_asset = asset_mask[:, :-1]
        current_asset = asset_mask[:, 1:]
        self.asset_stay_numerator += float((previous_asset & current_asset).sum())
        self.asset_stay_denominator += float(previous_asset.sum())
        self.asset_enter_numerator += float((~previous_asset & current_asset).sum())
        self.asset_enter_denominator += float((~previous_asset).sum())

        valid_pairs = feature_mask[:, :-1, :] & feature_mask[:, 1:, :]
        left = X[:, :-1, :]
        right = X[:, 1:, :]
        left_valid = np.where(valid_pairs, left, 0.0)
        right_valid = np.where(valid_pairs, right, 0.0)
        self.lag_count += valid_pairs.sum(axis=(0, 1))
        self.lag_sum_x += left_valid.sum(axis=(0, 1))
        self.lag_sum_y += right_valid.sum(axis=(0, 1))
        self.lag_sum_xx += (left_valid * left_valid).sum(axis=(0, 1))
        self.lag_sum_yy += (right_valid * right_valid).sum(axis=(0, 1))
        self.lag_sum_xy += (left_valid * right_valid).sum(axis=(0, 1))

        rows = X[asset_mask]
        if rows.shape[0] > 2048:
            rows = rows[rng.choice(rows.shape[0], size=2048, replace=False)]
        rows = rows.astype(np.float64, copy=False)
        self.corr_count += rows.shape[0]
        self.corr_sum += rows.sum(axis=0)
        self.corr_outer += rows.T @ rows

    def save(self, path: Path, feature_names: Sequence[str]) -> None:
        """由累计矩计算相关系数，并按固定特征顺序保存校准文件。"""

        observation_rate = self.feature_observed / max(self.feature_possible, 1.0)
        n = np.maximum(self.lag_count, 1.0)
        covariance = self.lag_sum_xy - self.lag_sum_x * self.lag_sum_y / n
        variance_x = self.lag_sum_xx - self.lag_sum_x * self.lag_sum_x / n
        variance_y = self.lag_sum_yy - self.lag_sum_y * self.lag_sum_y / n
        lag_corr = covariance / np.sqrt(np.maximum(variance_x * variance_y, 1e-12))
        lag_corr = np.nan_to_num(lag_corr, nan=0.70, posinf=0.95, neginf=0.0)
        lag_corr = np.clip(lag_corr, -0.95, 0.98)

        if self.corr_count > 1:
            covariance_matrix = (
                self.corr_outer - np.outer(self.corr_sum, self.corr_sum) / self.corr_count
            ) / (self.corr_count - 1)
            scale = np.sqrt(np.maximum(np.diag(covariance_matrix), 1e-12))
            correlation = covariance_matrix / np.outer(scale, scale)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(correlation, 1.0)
        else:
            correlation = np.eye(self.num_features, dtype=np.float64)

        # 校准量不是训练张量，因此使用可读 JSON 而不额外创建 NPZ。
        atomic_json(
            path,
            {
                "format_version": FORMAT_VERSION,
                "feature_names": list(feature_names),
                "feature_observation_rate": observation_rate.astype(np.float32).tolist(),
                "lag1_correlation": lag_corr.astype(np.float32).tolist(),
                "cross_feature_correlation": correlation.astype(np.float32).tolist(),
                "asset_observation_rate": float(
                    self.asset_observed / max(self.asset_total, 1.0)
                ),
                "asset_stay_probability": float(
                    self.asset_stay_numerator / max(self.asset_stay_denominator, 1.0)
                ),
                "asset_enter_probability": float(
                    self.asset_enter_numerator / max(self.asset_enter_denominator, 1.0)
                ),
            },
        )


def stack_and_save(path: Path, episodes: Sequence[Mapping[str, np.ndarray]]) -> None:
    """把同形状 episode 沿新的 E 维堆叠，原子保存为一个 NPZ 分片。"""

    keys = tuple(episodes[0].keys())
    if keys != TRAINING_ARRAY_NAMES:
        raise ValueError(
            f"training shard keys must be exactly {TRAINING_ARRAY_NAMES}, got {keys}"
        )
    if any(tuple(episode.keys()) != keys for episode in episodes):
        raise ValueError("all episodes in one shard must have identical ordered keys")
    arrays = {key: np.stack([episode[key] for episode in episodes]) for key in keys}
    atomic_npz(path, **arrays)


def main() -> None:
    """运行缓存、时间切分、对齐抽样、Y 合成、分片和 manifest 保存。"""

    # 先验证参数并识别 94 个特征，防止长时间扫描后才发现配置错误。
    args = parse_args()
    # 统一转为绝对路径，使日志、manifest 和实际写入位置完全一致。
    args.input_csv = args.input_csv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    validate_args(args)
    print(f"Input CSV: {args.input_csv}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    features = feature_columns(args.input_csv)
    # 把长表转换为可复用的逐月缓存；相同源文件和参数会自动复用。
    cache_path, cache_manifest = build_month_cache(
        args.input_csv, args.cache_dir, features, args.chunksize, args.max_source_rows
    )
    all_dates = np.asarray(cache_manifest["dates"], dtype=np.int64)
    # 先按日期切分，再在各 split 内生成窗口，确保月份不会跨集合泄漏。
    split_dates = split_calendar(all_dates, args.train_fraction, args.validation_fraction)
    requested = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
        "test": args.test_episodes,
    }
    starts = {
        split: valid_window_starts(dates, args.time_steps)
        for split, dates in split_dates.items()
    }
    for split, count in requested.items():
        if count > 0 and starts[split].size == 0:
            raise ValueError(
                f"split {split!r} has no contiguous {args.time_steps}-month window"
            )

    # 为保护已有训练数据，本脚本拒绝写入非空输出目录。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; choose a new directory"
        )

    @lru_cache(maxsize=max(2 * args.time_steps, 64))
    def load_month(date_value: int) -> Tuple[np.ndarray, np.ndarray]:
        """读取一个月的 permno 和原始特征，并关闭底层 NPZ 文件句柄。"""

        with np.load(cache_path / f"month_{date_value}.npz", allow_pickle=False) as data:
            return data["stock_ids"].copy(), data["X_obs"].copy()

    # 为三个 split 派生互不重叠、但可由根 seed 完全复现的随机流。
    root_seed = np.random.SeedSequence(args.seed)
    split_seed_sequences = root_seed.spawn(len(requested))
    metadata_path = args.output_dir / "generation_metadata.jsonl.gz"
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    shard_files: List[str] = []
    total_episodes = 0
    calibration = CalibrationAccumulator(len(features))
    parent_selection_counts_by_split: Dict[str, List[int]] = {}

    with gzip.open(metadata_temporary, "wt", encoding="utf-8", newline="\n") as metadata_file:
        for (split, episode_count), split_seed in zip(requested.items(), split_seed_sequences):
            split_directory = args.output_dir / split
            split_directory.mkdir(parents=True, exist_ok=True)
            split_rng = np.random.default_rng(split_seed)
            # 每个 split 独立均衡，避免验证/测试标签分布依赖训练集生成数量。
            parent_selection_counts = np.zeros(len(features), dtype=np.int64)
            pending: List[Dict[str, np.ndarray]] = []
            shard_index = 0
            for local_index in range(episode_count):
                # 每个 episode 独立保存 seed，之后无需保存随机状态也能重建。
                episode_seed = int(split_rng.integers(0, np.iinfo(np.uint32).max))
                rng = np.random.default_rng(episode_seed)
                start = int(rng.choice(starts[split]))
                dates = split_dates[split][start : start + args.time_steps]
                month_data = [load_month(int(date)) for date in dates]
                all_ids = np.concatenate([item[0] for item in month_data])
                # 资格规则只依赖股票覆盖率，不依赖随后抽取的目标父节点。
                unique_ids, counts = np.unique(all_ids, return_counts=True)
                minimum_count = int(math.ceil(args.min_asset_coverage * args.time_steps))
                eligible = unique_ids[counts >= minimum_count]
                if eligible.size == 0:
                    raise RuntimeError(
                        f"no stock satisfies coverage={args.min_asset_coverage} in window {dates[0]}-{dates[-1]}"
                    )
                selected_count = min(args.num_assets, eligible.size)
                selected = np.sort(rng.choice(eligible, size=selected_count, replace=False))
                stock_ids = np.full(args.num_assets, -1, dtype=np.int64)
                stock_ids[:selected_count] = selected
                # 稠密化为 [N,T,D]：不存在的股票月和特征缺失暂时都为 NaN。
                X_obs = np.full(
                    (args.num_assets, args.time_steps, len(features)), np.nan, dtype=np.float32
                )
                asset_mask = np.zeros((args.num_assets, args.time_steps), dtype=bool)
                for time_index, (month_ids, month_values) in enumerate(month_data):
                    positions = np.searchsorted(month_ids, selected)
                    present = positions < month_ids.size
                    present[present] &= month_ids[positions[present]] == selected[present]
                    target_rows = np.flatnonzero(present)
                    source_rows = positions[present]
                    asset_mask[target_rows, time_index] = True
                    X_obs[target_rows, time_index, :] = month_values[source_rows]

                # 截面秩变换并区分“股票月不存在”和“具体特征缺失”。
                X, feature_mask = rank_transform(X_obs, asset_mask)
                # 真实 X 不改动；这里只合成目标方程及已知的 z/tau 监督。
                target_arrays, target_metadata = synthesize_target(
                    X, asset_mask, feature_mask, rng,
                    args.min_parents, args.max_parents,
                    args.min_parent_observations,
                    args.min_parent_observation_rate,
                    parent_selection_counts,
                    args.snr_low, args.snr_high,
                    args.hard_negative_threshold,
                )
                episode_arrays: Dict[str, np.ndarray] = {
                    "X": X.astype(np.float32, copy=False),
                    "Y": target_arrays["Y"],
                    "z": target_arrays["z"],
                    "tau_direct": target_arrays["tau_direct"],
                    "asset_mask": asset_mask,
                    "feature_mask": feature_mask,
                    "parent_candidate_mask": target_arrays["parent_candidate_mask"],
                    "target_mask": asset_mask.copy(),
                    "time_padding_mask": ~asset_mask.any(axis=0),
                }
                calibration.update(X, asset_mask, feature_mask, rng)
                position_in_shard = len(pending)
                pending.append(episode_arrays)
                metadata = {
                    "episode_id": total_episodes,
                    "source_type": "semi_synthetic",
                    "split": split,
                    "shard": f"{split}/shard_{shard_index:05d}.npz",
                    "position_in_shard": position_in_shard,
                    "episode_seed": episode_seed,
                    "window_start": int(dates[0]),
                    "window_end": int(dates[-1]),
                    "num_real_assets": selected_count,
                    "market_state_dim": 0,
                    # 以下内容只用于复现、审计或额外消融，不进入训练 NPZ。
                    "stock_ids": stock_ids.tolist(),
                    "dates": dates.astype(np.int32).tolist(),
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
                metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                total_episodes += 1

                is_last = local_index == episode_count - 1
                # 达到分片容量或 split 末尾时统一堆叠落盘。
                if len(pending) == args.episodes_per_shard or is_last:
                    shard_relative = Path(split) / f"shard_{shard_index:05d}.npz"
                    stack_and_save(args.output_dir / shard_relative, pending)
                    shard_files.append(str(shard_relative))
                    pending = []
                    shard_index += 1
            parent_selection_counts_by_split[split] = parent_selection_counts.tolist()

    # 只在压缩元数据完整关闭后替换最终文件，避免生成中断留下可误读文件。
    os.replace(metadata_temporary, metadata_path)

    # 保存从本次真实 episode 估计的分布摘要，供全合成脚本直接复用。
    calibration_path = args.output_dir / "calibration.json"
    calibration.save(calibration_path, features)
    # manifest 是训练加载器理解目录的唯一入口，也明确声明当前 K=0。
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset_name": "GKX DataShare semi-synthetic CausalRank episodes",
        "source_type": "semi_synthetic",
        "source_csv": str(args.input_csv.resolve()),
        "source_cache": str(cache_path.resolve()),
        "feature_names": features,
        "dimensions": {"N": args.num_assets, "T": args.time_steps, "D": len(features)},
        "market_state_dim": 0,
        "contains_C": False,
        "C_handling": "Pass C=None and construct the model with market_state_dim=0.",
        "mask_semantics": {
            "asset_mask": "1 iff the stock-month row exists; 0 also marks padded assets",
            "feature_mask": "1 iff the stock-month exists and that characteristic was observed",
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
        "preprocessing": "monthly cross-sectional average rank mapped to [-1,1]; missing values are 0 with masks retained",
        "split_date_ranges": {
            split: ([int(values[0]), int(values[-1])] if len(values) else None)
            for split, values in split_dates.items()
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
        "calibration_file": "calibration.json",
        "readme": "README.md",
        "seed": args.seed,
        "generation_arguments": vars(args) | {
            "input_csv": str(args.input_csv),
            "output_dir": str(args.output_dir),
            "cache_dir": str(args.cache_dir),
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    write_dataset_readme(args.output_dir, manifest)
    print(f"Generated {total_episodes} semi-synthetic episodes in {args.output_dir}")
    print(f"Shape per episode: N={args.num_assets}, T={args.time_steps}, D={len(features)}, K=0")
    print("Current GKX DataShare has no C; shards intentionally omit a C array.")


if __name__ == "__main__":
    main()
