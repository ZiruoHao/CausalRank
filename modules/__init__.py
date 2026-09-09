"""CausalRank 的模型与训练数据公共接口。"""

# Dataset 已移动到 modules；统一从本包导入，避免训练入口依赖旧目录。
from .causal_rank_dataset import (
    CausalRankDataset,
    TRAINING_ARRAY_NAMES,
    TRAINING_ARRAY_NAMES_V1,
    create_causal_rank_dataloader,
    derive_factor_observation_mask,
)
# 将面板编码器暴露在 modules 包的顶层，方便外部直接导入。
from .causal_rank_panel_encoder import CausalRankPanelEncoder
# 将最终因果排序解码器暴露在包顶层，供训练和推理统一调用。
from .causal_ranking_decoder import CausalRankingDecoder
# 将目标感知父节点交互及其显式条件证据编码器暴露在包顶层，便于完整模型
# 组合和独立诊断岭回归/偏相关统计，而不依赖模块内部路径。
from .target_aware_parent_interaction import (
    ConditionalIncrementalEvidenceEncoder,
    TargetAwareParentInteraction,
)

# 明确声明本包希望对外公开的类，避免工具自动导出内部辅助组件。
__all__ = [
    "CausalRankDataset",
    "CausalRankPanelEncoder",
    "CausalRankingDecoder",
    "ConditionalIncrementalEvidenceEncoder",
    "TRAINING_ARRAY_NAMES",
    "TRAINING_ARRAY_NAMES_V1",
    "TargetAwareParentInteraction",
    "create_causal_rank_dataloader",
    "derive_factor_observation_mask",
]
