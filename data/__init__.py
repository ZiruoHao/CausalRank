"""旧数据包兼容入口；新代码应从 :mod:`modules` 导入 Dataset。"""

if __package__ and "." in __package__:
    from ..modules.causal_rank_dataset import (
        CausalRankDataset,
        TRAINING_ARRAY_NAMES,
        TRAINING_ARRAY_NAMES_V1,
        create_causal_rank_dataloader,
        derive_factor_observation_mask,
    )
else:
    from modules.causal_rank_dataset import (
        CausalRankDataset,
        TRAINING_ARRAY_NAMES,
        TRAINING_ARRAY_NAMES_V1,
        create_causal_rank_dataloader,
        derive_factor_observation_mask,
    )

__all__ = [
    "CausalRankDataset",
    "TRAINING_ARRAY_NAMES",
    "TRAINING_ARRAY_NAMES_V1",
    "create_causal_rank_dataloader",
    "derive_factor_observation_mask",
]
