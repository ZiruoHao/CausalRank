"""CausalRank 合成与半合成训练分片的 PyTorch Dataset。

两个数据生成器共享同一份训练协议。v2 NPZ shard 包含 E 个 episode：

    X:                 float32 [E,N,T,D]
    Y:                 float32 [E,N,T]
    z:                 bool    [E,D]
    tau_direct:        float32 [E,D]
    asset_mask:        bool    [E,N,T]
    feature_mask:      bool    [E,N,T,D]
    parent_candidate_mask: bool [E,D]
    target_mask:       bool    [E,N,T]
    time_padding_mask: bool    [E,T]

本 Dataset 不在初始化时把整个数据集读入内存。它只读取每个 shard 中很小的
z 数组建立全局索引，实际访问 episode 时才解压对应 shard，并通过 LRU 缓存
保留最近使用的少量分片。生产训练必须存在 manifest.json；生成中断后留下的
临时目录会被明确拒绝，避免把不完整数据误用于训练。

部分 episode 可能存在整个历史窗口都未观测到的因子。Dataset 保留原始字段
协议；训练时可用 ``derive_factor_observation_mask(feature_mask)`` 得到 [D] 或
[B,D] 的监督有效性掩码，并同时传给父节点交互模块与逐因子损失。
"""

from __future__ import annotations

# argparse 提供直接检查真实数据目录的命令行入口。
import argparse
# bisect 把全局 episode 下标映射到 shard 内部下标。
import bisect
# json 读取生成器保存的数据协议和分片列表。
import json
# OrderedDict 实现每个 DataLoader worker 独立的 LRU shard 缓存。
from collections import OrderedDict
# Path 用于安全解析 manifest 中的相对分片路径。
from pathlib import Path
# 类型标注明确 Dataset 返回值和内部缓存结构。
from typing import Any, Dict, List, Mapping, Optional, Tuple

# NumPy 负责读取压缩 NPZ 和执行协议验证。
import numpy as np
# torch.from_numpy 避免不必要的数据复制。
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# v1 没有显式的父候选监督掩码；读取时由 feature_mask 安全退化。
TRAINING_ARRAY_NAMES_V1: Tuple[str, ...] = (
    "X",
    "Y",
    "z",
    "tau_direct",
    "asset_mask",
    "feature_mask",
    "target_mask",
    "time_padding_mask",
)
# 当前两个生成器写出的 v2 协议。
TRAINING_ARRAY_NAMES: Tuple[str, ...] = (
    "X",
    "Y",
    "z",
    "tau_direct",
    "asset_mask",
    "feature_mask",
    "parent_candidate_mask",
    "target_mask",
    "time_padding_mask",
)
SUPPORTED_FORMAT_VERSIONS = {1, 2}
# 每个字段在落盘时必须使用的精确 dtype。
EXPECTED_DTYPES: Mapping[str, np.dtype] = {
    "X": np.dtype(np.float32),
    "Y": np.dtype(np.float32),
    "z": np.dtype(np.bool_),
    "tau_direct": np.dtype(np.float32),
    "asset_mask": np.dtype(np.bool_),
    "feature_mask": np.dtype(np.bool_),
    "parent_candidate_mask": np.dtype(np.bool_),
    "target_mask": np.dtype(np.bool_),
    "time_padding_mask": np.dtype(np.bool_),
}


def derive_factor_observation_mask(feature_mask: Tensor) -> Tensor:
    """从 feature_mask 派生 episode 级因子可观测性。

    输入可为单 episode 的 ``[N,T,D]`` 或 batch 的 ``[B,N,T,D]``，输出分别
    为 ``[D]`` 或 ``[B,D]``。True 表示该因子在相应 episode 中至少有一个
    真实观测，因此可以参与候选竞争和监督损失。
    """

    if feature_mask.ndim not in (3, 4):
        raise ValueError("feature_mask 必须是 [N,T,D] 或 [B,N,T,D]。")
    if feature_mask.dtype != torch.bool:
        raise TypeError("feature_mask 必须是布尔张量。")
    observation_dimensions = (0, 1) if feature_mask.ndim == 3 else (1, 2)
    return feature_mask.any(dim=observation_dimensions)


class CausalRankDataset(Dataset):
    """按 episode 懒加载一个 split 的 CausalRank NPZ 分片。

    参数：
        dataset_directory: 包含 manifest.json 和 train/validation/test 的目录。
        split: 需要加载的 train、validation 或 test。
        max_cached_shards: 每个进程最多在内存中保留的已解压分片数量。
        validate_shards: 首次加载分片时是否执行完整数值与掩码一致性验证。
    """

    def __init__(
        self,
        dataset_directory: Path,
        split: str = "train",
        max_cached_shards: int = 2,
        validate_shards: bool = True,
    ) -> None:
        """读取 manifest、检查分片列表并建立全局 episode 索引。"""

        super().__init__()
        if split not in {"train", "validation", "test"}:
            raise ValueError("split 必须是 train、validation 或 test。")
        if max_cached_shards <= 0:
            raise ValueError("max_cached_shards 必须为正整数。")

        self.dataset_directory = Path(dataset_directory).expanduser().resolve()
        self.split = split
        self.max_cached_shards = max_cached_shards
        self.validate_shards = validate_shards
        self._cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()

        manifest_path = self.dataset_directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"找不到 {manifest_path}。数据生成可能尚未完成；"
                "请勿直接训练只含 shard 或 .tmp 文件的中断目录。"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest: Dict[str, Any] = json.load(handle)
        self._validate_manifest()
        self.format_version = int(self.manifest["format_version"])
        self.training_array_names = (
            TRAINING_ARRAY_NAMES_V1
            if self.format_version == 1
            else TRAINING_ARRAY_NAMES
        )

        dimensions = self.manifest["dimensions"]
        self.num_assets = int(dimensions["N"])
        self.num_times = int(dimensions["T"])
        self.num_factors = int(dimensions["D"])
        self.market_state_dim = int(self.manifest["market_state_dim"])
        self.source_type = str(self.manifest["source_type"])

        # manifest 中的路径必须属于当前 split，且不能逃逸数据集根目录。
        root = self.dataset_directory
        shard_paths: List[Path] = []
        for relative_name in self.manifest["shards"]:
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"manifest 包含不安全的 shard 路径：{relative_name}")
            if not relative_path.parts or relative_path.parts[0] != split:
                continue
            shard_path = (root / relative_path).resolve()
            try:
                shard_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"shard 路径位于数据集目录之外：{relative_name}"
                ) from error
            if not shard_path.is_file():
                raise FileNotFoundError(f"manifest 声明的 shard 不存在：{shard_path}")
            shard_paths.append(shard_path)
        if not shard_paths:
            raise ValueError(f"数据集没有可用于 split={split} 的 shard。")
        self.shard_paths: Tuple[Path, ...] = tuple(shard_paths)

        # 每个 shard 的 z 很小；只读取它即可获得 episode 数而无需解压大 X。
        self._shard_episode_counts: List[int] = []
        self._cumulative_episode_counts: List[int] = []
        running_total = 0
        for shard_path in self.shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard:
                if tuple(shard.files) != self.training_array_names:
                    raise ValueError(
                        f"{shard_path} 字段应为 {self.training_array_names}，"
                        f"实际为 {tuple(shard.files)}。"
                    )
                labels = shard["z"]
                if labels.ndim != 2 or labels.shape[1] != self.num_factors:
                    raise ValueError(f"{shard_path} 的 z 形状不符合 [E,D]。")
                episode_count = int(labels.shape[0])
                if episode_count <= 0:
                    raise ValueError(f"{shard_path} 不包含任何 episode。")
            self._shard_episode_counts.append(episode_count)
            running_total += episode_count
            self._cumulative_episode_counts.append(running_total)

        expected_count = int(self.manifest["episode_counts"].get(split, 0))
        if running_total != expected_count:
            raise ValueError(
                f"split={split} 的 manifest 数量为 {expected_count}，"
                f"但 shard 中实际共有 {running_total} 个 episode。"
            )

    def _validate_manifest(self) -> None:
        """验证训练所需的 manifest 字段与当前 Dataset 协议兼容。"""

        required_fields = {
            "format_version",
            "source_type",
            "dimensions",
            "market_state_dim",
            "contains_C",
            "episode_counts",
            "shards",
            "training_arrays",
        }
        missing = required_fields.difference(self.manifest)
        if missing:
            raise ValueError(f"manifest 缺少字段：{sorted(missing)}")
        format_version = int(self.manifest["format_version"])
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"不支持 format_version={self.manifest['format_version']}，"
                f"当前支持 {sorted(SUPPORTED_FORMAT_VERSIONS)}。"
            )
        dimensions = self.manifest["dimensions"]
        if not all(int(dimensions.get(name, 0)) > 0 for name in ("N", "T", "D")):
            raise ValueError("manifest dimensions 必须包含正数 N、T、D。")
        if int(self.manifest["market_state_dim"]) != 0:
            raise ValueError("当前 Dataset 协议尚未包含 C 数组，只支持 market_state_dim=0。")
        if bool(self.manifest["contains_C"]):
            raise ValueError("当前八字段训练协议不应声明 contains_C=True。")
        expected_names = (
            TRAINING_ARRAY_NAMES_V1 if format_version == 1 else TRAINING_ARRAY_NAMES
        )
        if tuple(self.manifest["training_arrays"].keys()) != expected_names:
            raise ValueError(
                "manifest training_arrays 与当前八字段训练协议不一致。"
            )
        if not isinstance(self.manifest["shards"], list):
            raise TypeError("manifest shards 必须是路径列表。")

    def _expected_shapes(self, episode_count: int) -> Mapping[str, Tuple[int, ...]]:
        """返回一个 shard 中八个数组的理论形状。"""

        batch = episode_count
        assets = self.num_assets
        times = self.num_times
        factors = self.num_factors
        return {
            "X": (batch, assets, times, factors),
            "Y": (batch, assets, times),
            "z": (batch, factors),
            "tau_direct": (batch, factors),
            "asset_mask": (batch, assets, times),
            "feature_mask": (batch, assets, times, factors),
            "parent_candidate_mask": (batch, factors),
            "target_mask": (batch, assets, times),
            "time_padding_mask": (batch, times),
        }

    def _validate_loaded_shard(
        self,
        shard_path: Path,
        arrays: Mapping[str, np.ndarray],
        expected_count: int,
    ) -> None:
        """验证形状、dtype、有限值及掩码之间的逻辑约束。"""

        expected_shapes = self._expected_shapes(expected_count)
        for name in self.training_array_names:
            array = arrays[name]
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"{shard_path} 中 {name} 应为 {expected_shapes[name]}，"
                    f"实际为 {array.shape}。"
                )
            if array.dtype != EXPECTED_DTYPES[name]:
                raise TypeError(
                    f"{shard_path} 中 {name} 应为 {EXPECTED_DTYPES[name]}，"
                    f"实际为 {array.dtype}。"
                )

        factors = arrays["X"]
        targets = arrays["Y"]
        labels = arrays["z"]
        effects = arrays["tau_direct"]
        asset_mask = arrays["asset_mask"]
        feature_mask = arrays["feature_mask"]
        parent_candidate_mask = arrays.get("parent_candidate_mask")
        target_mask = arrays["target_mask"]
        time_padding_mask = arrays["time_padding_mask"]

        if not np.isfinite(factors).all():
            raise ValueError(f"{shard_path} 的 X 包含 NaN 或 Inf。")
        if not np.isfinite(targets).all():
            raise ValueError(f"{shard_path} 的 Y 包含 NaN 或 Inf。")
        if not np.isfinite(effects).all() or np.any(effects < 0.0):
            raise ValueError(f"{shard_path} 的 tau_direct 必须是有限非负数。")
        if np.any(feature_mask & ~asset_mask[..., None]):
            raise ValueError(f"{shard_path} 的 feature_mask 越过 asset_mask。")
        if np.any(target_mask & ~asset_mask):
            raise ValueError(f"{shard_path} 的 target_mask 越过 asset_mask。")
        if parent_candidate_mask is not None:
            observed_factors = feature_mask.any(axis=(1, 2))
            if np.any(parent_candidate_mask & ~observed_factors):
                raise ValueError(f"{shard_path} 的父候选因子必须至少有一个观测。")
            if np.any(labels & ~parent_candidate_mask):
                raise ValueError(f"{shard_path} 的直接父节点不在 parent_candidate_mask 内。")
        expected_time_mask = ~asset_mask.any(axis=1)
        if not np.array_equal(time_padding_mask, expected_time_mask):
            raise ValueError(f"{shard_path} 的 time_padding_mask 与 asset_mask 不一致。")
        if np.any(time_padding_mask.all(axis=1)):
            raise ValueError(f"{shard_path} 包含完全没有有效时间点的 episode。")
        if np.any(factors[~feature_mask] != 0.0):
            raise ValueError(f"{shard_path} 的 X 缺失位置必须填充为 0。")
        if np.any(targets[~target_mask] != 0.0):
            raise ValueError(f"{shard_path} 的 Y 无效位置必须填充为 0。")
        if np.any(effects[~labels] != 0.0):
            raise ValueError(f"{shard_path} 的非父节点 tau_direct 必须为 0。")
        if np.any(labels.sum(axis=1) == 0):
            raise ValueError(f"{shard_path} 包含没有任何直接父节点的 episode。")

    def _load_shard(self, shard_index: int) -> Dict[str, np.ndarray]:
        """读取、验证并缓存一个完整 shard。"""

        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            # 移到 OrderedDict 末尾，表示它刚被访问。
            self._cache[shard_index] = cached
            return cached

        shard_path = self.shard_paths[shard_index]
        with np.load(shard_path, allow_pickle=False) as shard:
            if tuple(shard.files) != self.training_array_names:
                raise ValueError(f"{shard_path} 的字段在 Dataset 初始化后发生变化。")
            arrays = {
                name: shard[name]
                for name in self.training_array_names
            }
        if self.validate_shards:
            self._validate_loaded_shard(
                shard_path,
                arrays,
                self._shard_episode_counts[shard_index],
            )
        self._cache[shard_index] = arrays
        # 超过上限时删除最久未使用的分片，释放大数组内存。
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return arrays

    def __len__(self) -> int:
        """返回当前 split 的 episode 总数。"""

        return self._cumulative_episode_counts[-1]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """返回一个 episode，并保留生成器的八个原始字段名。"""

        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"episode index {index} 超出 [0,{length})。")

        shard_index = bisect.bisect_right(
            self._cumulative_episode_counts,
            index,
        )
        previous_total = (
            0 if shard_index == 0
            else self._cumulative_episode_counts[shard_index - 1]
        )
        local_index = index - previous_total
        arrays = self._load_shard(shard_index)
        # 默认 DataLoader collate 会把这些单 episode Tensor 堆叠为 [B,...]。
        return {
            name: torch.from_numpy(arrays[name][local_index])
            for name in self.training_array_names
        }

    def clear_cache(self) -> None:
        """显式释放当前进程缓存的已解压 shard。"""

        self._cache.clear()

    def __getstate__(self) -> Dict[str, Any]:
        """DataLoader 创建 worker 时不复制父进程中已经缓存的大数组。"""

        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


def create_causal_rank_dataloader(
    dataset_directory: Path,
    split: str,
    batch_size: int,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_cached_shards: int = 2,
    validate_shards: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """构造适配固定 N/T/D episode 的标准 DataLoader。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数。")
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数。")
    dataset = CausalRankDataset(
        dataset_directory=dataset_directory,
        split=split,
        max_cached_shards=max_cached_shards,
        validate_shards=validate_shards,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造真实数据目录的 Dataset 自检参数。"""

    parser = argparse.ArgumentParser(description="检查 CausalRank 训练 Dataset")
    parser.add_argument("dataset_directory", type=Path, help="包含 manifest.json 的目录")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    """加载一个真实 batch 并打印八个字段的形状与 dtype。"""

    args = _build_argument_parser().parse_args()
    loader = create_causal_rank_dataloader(
        dataset_directory=args.dataset_directory,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    dataset = loader.dataset
    print(
        f"CausalRankDataset 测试通过：split={args.split}, "
        f"episodes={len(dataset)}, source={dataset.source_type}"
    )
    for name in dataset.training_array_names:
        value = batch[name]
        print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")
    factor_mask = derive_factor_observation_mask(batch["feature_mask"])
    print(
        "factor_observation_mask: "
        f"shape={tuple(factor_mask.shape)}, valid={int(factor_mask.sum())}"
    )


if __name__ == "__main__":
    main()
