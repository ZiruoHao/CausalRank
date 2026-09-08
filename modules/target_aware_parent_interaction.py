from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class TargetAwareParentInteraction(nn.Module):
    """
    CausalRank 的目标感知父节点交互模块。

    输入：
        factor_features: [B, D, E] 或 [D, E]
        target_feature:  [B, E]    或 [E]
        factor_mask:     [B, D]    或 [D]，True 表示该 episode 中因子可观测

    输出：
        parent_features: [B, D, E] 或 [D, E]

    核心流程：

        H, h_Y
        ↓
        Target Relation Extraction
        ↓
        R^Y = (r_1^Y, ..., r_D^Y)
        ↓
        Target-Conditioned Leave-One-Out Competition
        ↓
        C = (c_1, ..., c_D)
        ↓
        Parent Evidence Fusion
        ↓
        E = (e_1, ..., e_D)

    其中对于候选因子 X_j：

        r_j^Y
            表示 X_j 相对于目标 Y 的关系证据

        c_j
            表示其他候选变量 H_{-j} 中，与目标 Y 相关并且
            能与 X_j 形成竞争关系的上下文

        e_j
            表示经过候选变量竞争后得到的 parent-aware representation

    注意：
        本模块并不执行形式化的条件独立检验。
        它通过合成因果监督学习区分直接父节点和高度相关非父节点。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须为正整数。")

        if num_heads <= 0:
            raise ValueError("num_heads 必须为正整数。")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须为正整数。")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim 必须能够被 num_heads 整除。"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout 必须满足 0 <= dropout < 1。"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # ============================================================
        # 1. Target Relation Extraction
        # ============================================================

        # 候选因子和目标分别映射到共享关系空间
        self.factor_query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.target_key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.target_value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # 根据 factor-target 双线性交互产生逐维目标门控
        self.target_gate = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.Sigmoid(),
        )

        # 输入包括：
        #
        # h_j
        # h_Y
        # h_j - h_Y
        # h_j * h_Y
        # gated target
        #
        # 共 5E 维
        self.target_interaction_network = nn.Sequential(
            nn.LayerNorm(embedding_dim * 5),
            nn.Linear(
                embedding_dim * 5,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
            nn.Dropout(dropout),
        )

        self.target_interaction_norm = nn.LayerNorm(
            embedding_dim
        )

        # ============================================================
        # 2. Target Relevance Gate
        # ============================================================

        # 对已经 target-conditioned 的候选表示估计
        # “当前候选因子与目标 Y 有多强的关系证据”
        #
        # 这个 gate 不代表因果概率，只用于构造竞争上下文。
        self.target_relevance_gate = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(
                embedding_dim,
                1,
            ),
            nn.Sigmoid(),
        )

        # ============================================================
        # 3. Target-Conditioned Conditional Competition
        # ============================================================

        # 与旧版本不同：
        #
        # query/key/value 不再来自原始 H，
        # 而来自已经经过 Y 条件化的 R^Y。
        #
        # 因此竞争发生在：
        #
        # “哪些候选变量同时表现出与 Y 有关的证据”
        #
        # 之间。
        self.conditional_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.conditional_attention_norm = nn.LayerNorm(
            embedding_dim
        )

        self.conditional_feedforward = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
            nn.Dropout(dropout),
        )

        self.conditional_output_norm = nn.LayerNorm(
            embedding_dim
        )

        # ============================================================
        # 4. Parent Evidence Fusion
        # ============================================================

        # 比较：
        #
        # r_j^Y
        # c_j
        # r_j^Y - c_j
        # r_j^Y * c_j
        #
        # 学习当前候选因子相对于其他 target-related
        # 候选变量的父节点证据。
        self.parent_representation_network = nn.Sequential(
            nn.LayerNorm(
                embedding_dim * 4
            ),
            nn.Linear(
                embedding_dim * 4,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
            nn.Dropout(dropout),
        )

        self.parent_output_norm = nn.LayerNorm(
            embedding_dim
        )

    # ================================================================
    # 输入检查
    # ================================================================

    def _validate_and_batch_inputs(
        self,
        factor_features: Tensor,
        target_feature: Tensor,
        factor_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, bool]:

        remove_batch_dimension = (
            factor_features.ndim == 2
        )

        if factor_features.ndim not in (2, 3):
            raise ValueError(
                "factor_features 必须是 [D,E] 或 [B,D,E]。"
            )

        if remove_batch_dimension:

            if target_feature.ndim != 1:
                raise ValueError(
                    "当 factor_features 为 [D,E] 时，"
                    "target_feature 必须为 [E]。"
                )

            factor_features = factor_features.unsqueeze(0)
            target_feature = target_feature.unsqueeze(0)
            if factor_mask is not None:
                if factor_mask.ndim != 1:
                    raise ValueError(
                        "无 batch 输入的 factor_mask 必须为 [D]。"
                    )
                factor_mask = factor_mask.unsqueeze(0)

        elif target_feature.ndim != 2:

            raise ValueError(
                "当 factor_features 为 [B,D,E] 时，"
                "target_feature 必须为 [B,E]。"
            )

        batch_size, num_factors, embedding_dim = (
            factor_features.shape
        )

        if batch_size <= 0 or num_factors <= 0:
            raise ValueError(
                "batch size 和候选因子数量必须大于 0。"
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"factor_features 最后一维应为 "
                f"{self.embedding_dim}，"
                f"实际为 {embedding_dim}。"
            )

        if target_feature.shape != (
            batch_size,
            self.embedding_dim,
        ):
            raise ValueError(
                "target_feature 的维度必须为 [B,E]。"
            )

        # factor_mask 不是候选集合 padding；它只表示当前 episode 是否至少
        # 观测过该因子。缺省时保持原接口行为，把所有候选都视为可观测。
        if factor_mask is None:
            factor_mask = torch.ones(
                batch_size,
                num_factors,
                dtype=torch.bool,
                device=factor_features.device,
            )
        else:
            if factor_mask.shape != (batch_size, num_factors):
                raise ValueError("factor_mask 必须为 [B,D]。")
            if factor_mask.dtype != torch.bool:
                raise TypeError("factor_mask 必须是布尔张量，True 表示因子可观测。")
            if factor_mask.device != factor_features.device:
                raise ValueError("factor_mask 和 factor_features 必须位于同一设备。")

        if not torch.is_floating_point(
            factor_features
        ):
            raise TypeError(
                "factor_features 必须为浮点张量。"
            )

        if not torch.is_floating_point(
            target_feature
        ):
            raise TypeError(
                "target_feature 必须为浮点张量。"
            )

        if factor_features.device != target_feature.device:
            raise ValueError(
                "factor_features 与 target_feature "
                "必须位于同一设备。"
            )

        if factor_features.dtype != target_feature.dtype:
            raise TypeError(
                "factor_features 与 target_feature "
                "必须使用相同 dtype。"
            )

        if not torch.isfinite(
            factor_features
        ).all():
            raise ValueError(
                "factor_features 中存在 NaN 或 Inf。"
            )

        if not torch.isfinite(
            target_feature
        ).all():
            raise ValueError(
                "target_feature 中存在 NaN 或 Inf。"
            )

        return (
            factor_features,
            target_feature,
            factor_mask,
            remove_batch_dimension,
        )

    # ================================================================
    # Step 1
    # Target Relation Extraction
    # ================================================================

    def target_interaction(
        self,
        factor_features: Tensor,
        target_feature: Tensor,
    ) -> Tensor:
        """
        为每个候选因子提取相对于目标 Y 的关系表示。

        输入：
            factor_features [B,D,E]
            target_feature  [B,E]

        输出：
            target_features [B,D,E]
        """

        num_factors = factor_features.shape[1]

        expanded_target = (
            target_feature
            .unsqueeze(1)
            .expand(
                -1,
                num_factors,
                -1,
            )
        )

        # ------------------------------------------------------------
        # 双线性 factor-target interaction
        # ------------------------------------------------------------

        factor_query = (
            self.factor_query_projection(
                factor_features
            )
        )

        target_key = (
            self.target_key_projection(
                expanded_target
            )
        )

        target_value = (
            self.target_value_projection(
                expanded_target
            )
        )

        bilinear_interaction = (
            factor_query * target_key
        )

        # ------------------------------------------------------------
        # 每个候选因子决定目标表示中哪些维度最重要
        # ------------------------------------------------------------

        target_gate = self.target_gate(
            bilinear_interaction
        )

        gated_target = (
            target_gate * target_value
        )

        # ------------------------------------------------------------
        # 构造完整 target interaction
        # ------------------------------------------------------------

        interaction_input = torch.cat(
            (
                factor_features,
                expanded_target,
                factor_features
                - expanded_target,
                factor_features
                * expanded_target,
                gated_target,
            ),
            dim=-1,
        )

        interaction_update = (
            self.target_interaction_network(
                interaction_input
            )
        )

        target_features = (
            self.target_interaction_norm(
                factor_features
                + interaction_update
            )
        )

        return target_features

    # ================================================================
    # Leave-One-Out Mask
    # ================================================================

    def _build_leave_one_out_mask(
        self,
        num_factors: int,
        device: torch.device,
    ) -> Tensor:
        """
        对角线为 True。

        第 j 个候选因子无法在竞争分支中读取自身。
        """

        return torch.eye(
            num_factors,
            dtype=torch.bool,
            device=device,
        )

    # ================================================================
    # Step 2
    # Target-Conditioned Conditional Competition
    # ================================================================

    def conditional_context(
        self,
        factor_target_features: Tensor,
        factor_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        使用已经经过目标 Y 条件化的候选表示构造竞争上下文。

        对第 j 个候选：

            c_j =
                Attention(
                    r_j^Y,
                    R_{-j}^Y,
                    R_{-j}^Y
                )

        输出：
            [B,D,E]
        """

        batch_size, num_factors, _ = (
            factor_target_features.shape
        )

        if factor_mask is None:
            factor_mask = torch.ones(
                batch_size,
                num_factors,
                dtype=torch.bool,
                device=factor_target_features.device,
            )
        else:
            if factor_mask.shape != (batch_size, num_factors):
                raise ValueError("factor_mask 必须为 [B,D]。")
            if factor_mask.dtype != torch.bool:
                raise TypeError("factor_mask 必须是布尔张量。")
            if factor_mask.device != factor_target_features.device:
                raise ValueError(
                    "factor_mask 和 factor_target_features 必须位于同一设备。"
                )

        if num_factors == 1:

            # 没有其他候选因子时不存在竞争上下文
            return torch.zeros_like(
                factor_target_features
            )

        # ------------------------------------------------------------
        # 估计每个候选因子的 target relevance
        #
        # 注意：
        # 这里只用于控制其作为“竞争者”的强度，
        # 并不是 causal existence probability。
        # ------------------------------------------------------------

        relevance = (
            self.target_relevance_gate(
                factor_target_features
            )
        )

        # target relevance 较低的候选变量作为竞争者时影响更弱
        competition_tokens = (
            factor_target_features
            * relevance
        )

        diagonal_mask = (
            self._build_leave_one_out_mask(
                num_factors,
                factor_target_features.device,
            )
        )

        # ------------------------------------------------------------
        # Query：
        # 当前候选因子的 target-aware representation
        #
        # Key / Value：
        # 其他候选因子的 target-aware representation
        # ------------------------------------------------------------

        # 至少两个可观测因子的 episode 才能形成 leave-one-out 竞争。
        # 逐 batch 选择可避免“唯一可观测因子 + 对角掩码”形成全掩码注意力行。
        eligible_batch = factor_mask.sum(dim=1) >= 2
        if not eligible_batch.any():
            return torch.zeros_like(factor_target_features)

        selected_features = factor_target_features[eligible_batch]
        selected_tokens = competition_tokens[eligible_batch]
        selected_mask = factor_mask[eligible_batch]
        attended_context, _ = self.conditional_attention(
            query=selected_features,
            key=selected_tokens,
            value=selected_tokens,
            attn_mask=diagonal_mask,
            key_padding_mask=~selected_mask,
            need_weights=False,
        )

        hidden = self.conditional_attention_norm(attended_context)
        conditional_update = self.conditional_feedforward(hidden)
        selected_context = self.conditional_output_norm(
            hidden + conditional_update
        )
        # 不可观测 query 的输出不参与解码，也不能成为后续有效证据。
        selected_context = selected_context.masked_fill(
            ~selected_mask.unsqueeze(-1),
            0.0,
        )
        # 对不可形成竞争的 batch 保持严格零上下文，并把可用结果写回原顺序。
        eligible_indices = eligible_batch.nonzero(as_tuple=False).squeeze(1)
        return torch.zeros_like(factor_target_features).index_copy(
            0,
            eligible_indices,
            selected_context,
        )

    # ================================================================
    # Step 3
    # Parent Evidence Fusion
    # ================================================================

    def fuse(
        self,
        factor_target_features: Tensor,
        conditional_features: Tensor,
    ) -> Tensor:
        """
        比较候选因子的目标证据和其他变量形成的竞争上下文。

        对候选 X_j：

            r_j^Y
                当前变量的 target evidence

            c_j
                其他 target-related variables 的竞争证据

        网络利用：

            r_j^Y
            c_j
            r_j^Y - c_j
            r_j^Y * c_j

        学习 parent-aware representation。
        """

        competition_residual = (
            factor_target_features
            - conditional_features
        )

        competition_agreement = (
            factor_target_features
            * conditional_features
        )

        fusion_input = torch.cat(
            (
                factor_target_features,
                conditional_features,
                competition_residual,
                competition_agreement,
            ),
            dim=-1,
        )

        parent_update = (
            self.parent_representation_network(
                fusion_input
            )
        )

        # target evidence 作为主残差路径，
        # 条件竞争仅负责修正该证据。
        parent_features = (
            self.parent_output_norm(
                factor_target_features
                + parent_update
            )
        )

        return parent_features

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        factor_features: Tensor,
        target_feature: Tensor,
        factor_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        完整计算：

            H, h_Y

            ↓

            Target Relation Extraction

            ↓

            R^Y

            ↓

            Target-Conditioned
            Leave-One-Out Competition

            ↓

            C

            ↓

            Parent Evidence Fusion

            ↓

            E

        输出：
            [B,D,E]

        无 batch 输入时：
            [D,E]
        """

        (
            factor_features,
            target_feature,
            factor_mask,
            remove_batch_dimension,
        ) = self._validate_and_batch_inputs(
            factor_features,
            target_feature,
            factor_mask,
        )

        # Step 1
        factor_target_features = (
            self.target_interaction(
                factor_features,
                target_feature,
            )
        )
        # 完全未观测因子不能作为自身证据，也不能进入其他候选的条件上下文。
        factor_target_features = factor_target_features.masked_fill(
            ~factor_mask.unsqueeze(-1),
            0.0,
        )

        # Step 2
        #
        # 关键修改：
        #
        # 旧版本：
        # conditional_context(factor_features)
        #
        # 新版本：
        # conditional_context(factor_target_features)
        #
        # 即竞争发生在已经知道 Y 的表示空间中。
        conditional_features = (
            self.conditional_context(
                factor_target_features,
                factor_mask,
            )
        )

        # Step 3
        parent_features = self.fuse(
            factor_target_features,
            conditional_features,
        )
        # 融合网络含偏置，需再次归零不可观测位置，供外部监督掩码安全使用。
        parent_features = parent_features.masked_fill(
            ~factor_mask.unsqueeze(-1),
            0.0,
        )

        if remove_batch_dimension:
            parent_features = (
                parent_features.squeeze(0)
            )

        return parent_features
