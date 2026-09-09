from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class ConditionalIncrementalEvidenceEncoder(nn.Module):
    """从原始面板计算逐因子的显式条件增量证据。

    隐式 attention 能比较候选表示，但并不保证它会学出“控制其他 X 后，
    当前 X_j 是否仍解释 Y”这一统计量。本模块因此对每个 episode 构造一个
    掩码兼容的标准化设计矩阵，并计算两类互补证据：

    1. 全变量岭回归系数 beta_j，表示其他候选同时进入模型时 X_j 的增量；
    2. 从联合精度矩阵得到的偏相关，表示给定 X_{-j} 后 X_j 与 Y 的关系。

    这些量只由模型本来就收到的 X、Y 和观测掩码计算，不读取 z 或 tau，
    因而不是标签泄漏。后面的可训练投影只负责把六个低维统计量映射到与
    parent-aware representation 相同的 E 维空间。
    """

    statistic_dim = 6

    def __init__(
        self,
        embedding_dim: int,
        ridge: float = 1e-2,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须为正整数。")
        if ridge <= 0.0:
            raise ValueError("conditional ridge 必须为正数。")

        self.embedding_dim = embedding_dim
        self.ridge = float(ridge)
        # 统计量的物理尺度不同；先逐候选归一化，再用小型 MLP 形成条件证据
        # token。该投影是残差支路，不会替代上游学到的非线性面板表示。
        self.projection = nn.Sequential(
            nn.LayerNorm(self.statistic_dim),
            nn.Linear(self.statistic_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def compute_statistics(
        self,
        factors: Tensor,
        target_returns: Tensor,
        factor_mask: Tensor,
        feature_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """返回 `[B,D,6]` 的边际、岭回归、偏相关和覆盖率统计。

        缺失因子先在各自有效观测上中心化、标准化，再以零填充；由于零正好
        表示标准化后的均值，这相当于掩码感知的均值插补。随后按覆盖率缩放
        列范数，避免观测较少的候选仅因零值更多而被系统性压低。
        """

        if factors.ndim != 4:
            raise ValueError("factors 必须是 [B,N,T,D]。")
        if target_returns.ndim != 3:
            raise ValueError("target_returns 必须是 [B,N,T]。")
        batch_size, num_assets, num_times, num_factors = factors.shape
        if target_returns.shape != (batch_size, num_assets, num_times):
            raise ValueError("target_returns 的 [B,N,T] 必须与 factors 对齐。")
        if factor_mask.shape != (batch_size, num_factors):
            raise ValueError("factor_mask 必须是 [B,D]。")
        if factor_mask.dtype != torch.bool:
            raise TypeError("factor_mask 必须是布尔张量。")
        if factors.device != target_returns.device or factors.device != factor_mask.device:
            raise ValueError("条件证据的 factors、target_returns 和 mask 必须同设备。")
        if not torch.is_floating_point(factors) or not torch.is_floating_point(target_returns):
            raise TypeError("条件证据的 X 和 Y 必须为浮点张量。")
        if not torch.isfinite(factors).all() or not torch.isfinite(target_returns).all():
            raise ValueError("条件证据的 X 和 Y 不能包含 NaN 或 Inf。")

        expected_feature_shape = (batch_size, num_assets, num_times, num_factors)
        if feature_mask is None:
            feature_mask = torch.ones(
                expected_feature_shape,
                dtype=torch.bool,
                device=factors.device,
            )
        elif feature_mask.shape != expected_feature_shape or feature_mask.dtype != torch.bool:
            raise ValueError("feature_mask 必须是与 factors 同形状的布尔张量。")
        expected_target_shape = (batch_size, num_assets, num_times)
        if target_mask is None:
            target_mask = torch.ones(
                expected_target_shape,
                dtype=torch.bool,
                device=factors.device,
            )
        elif target_mask.shape != expected_target_shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask 必须是与 target_returns 同形状的布尔张量。")
        if feature_mask.device != factors.device or target_mask.device != factors.device:
            raise ValueError("feature_mask、target_mask 和 factors 必须同设备。")

        # 线性代数求解在 float16 下既不稳定也不被所有设备支持。显式关闭外层
        # AMP 并使用 float32；投影输出稍后再转换回主干表示 dtype。
        with torch.autocast(device_type=factors.device.type, enabled=False):
            x = factors.float().reshape(batch_size, num_assets * num_times, num_factors)
            y = target_returns.float().reshape(batch_size, num_assets * num_times)
            observed_x = feature_mask.reshape(
                batch_size, num_assets * num_times, num_factors
            )
            observed_y = target_mask.reshape(batch_size, num_assets * num_times)
            joint_observed = observed_x & observed_y.unsqueeze(-1)

            # 每列只使用 X_j 与 Y 同时有效的位置估计均值和方差。
            x_count = joint_observed.sum(dim=1).float()
            safe_x_count = x_count.clamp_min(1.0)
            x_mean = (x * joint_observed).sum(dim=1) / safe_x_count
            x_centered = (x - x_mean.unsqueeze(1)) * joint_observed
            x_variance = x_centered.square().sum(dim=1) / safe_x_count
            x_standardized = x_centered / torch.sqrt(
                x_variance.clamp_min(1e-6)
            ).unsqueeze(1)

            y_count = observed_y.sum(dim=1).float()
            safe_y_count = y_count.clamp_min(1.0)
            y_mean = (y * observed_y).sum(dim=1) / safe_y_count
            y_centered = (y - y_mean.unsqueeze(1)) * observed_y
            y_variance = y_centered.square().sum(dim=1) / safe_y_count
            y_standardized = y_centered / torch.sqrt(
                y_variance.clamp_min(1e-6)
            ).unsqueeze(1)

            # 均值插补会使缺失较多的列范数减小；除以 sqrt(coverage) 后，每个
            # 可用候选在完整目标样本尺度上的二阶矩仍约为 1。
            coverage = x_count / safe_y_count.unsqueeze(-1)
            x_design = x_standardized / torch.sqrt(
                coverage.clamp_min(1e-6)
            ).unsqueeze(1)
            usable = (
                factor_mask
                & (x_count >= 2.0)
                & (y_count >= 2.0).unsqueeze(-1)
                & (x_variance >= 1e-6)
                & (y_variance >= 1e-6).unsqueeze(-1)
            )
            x_design = x_design * usable.unsqueeze(1)

            denominator = safe_y_count.reshape(batch_size, 1, 1)
            gram = torch.bmm(x_design.transpose(1, 2), x_design) / denominator
            cross = torch.bmm(
                x_design.transpose(1, 2), y_standardized.unsqueeze(-1)
            ).squeeze(-1) / safe_y_count.unsqueeze(-1)

            # 岭回归同时放入所有候选。代理变量即便与 Y 高度相关，只要其信息
            # 已被真正父节点解释，对应 beta 就应接近 0。
            identity = torch.eye(
                num_factors, dtype=torch.float32, device=factors.device
            ).expand(batch_size, -1, -1)
            ridge_gram = gram + self.ridge * identity
            beta = torch.linalg.solve(ridge_gram, cross.unsqueeze(-1)).squeeze(-1)

            # 将 Y 拼到标准化设计矩阵末列，联合协方差的逆即精度矩阵。
            # precision[j,Y] 经对角缩放后得到控制其他全部 X 的偏相关。
            joint_design = torch.cat(
                (x_design, y_standardized.unsqueeze(-1)), dim=-1
            )
            joint_covariance = (
                torch.bmm(joint_design.transpose(1, 2), joint_design)
                / denominator
            )
            joint_identity = torch.eye(
                num_factors + 1,
                dtype=torch.float32,
                device=factors.device,
            ).expand(batch_size, -1, -1)
            cholesky = torch.linalg.cholesky(
                joint_covariance + self.ridge * joint_identity
            )
            precision = torch.cholesky_inverse(cholesky)
            precision_xy = precision[:, :num_factors, -1]
            precision_xx = precision[:, :num_factors, :num_factors].diagonal(
                dim1=-2, dim2=-1
            )
            precision_yy = precision[:, -1, -1].unsqueeze(-1)
            partial_correlation = -precision_xy / torch.sqrt(
                (precision_xx * precision_yy).clamp_min(1e-12)
            )

            # 六个通道分别保留符号、强度与数据可靠性；abs 通道对应当前
            # tau_direct 的非负强度语义，signed 通道仍可区分正负作用机制。
            statistics = torch.stack(
                (
                    cross,
                    beta,
                    beta.abs(),
                    partial_correlation,
                    partial_correlation.abs(),
                    coverage.clamp(0.0, 1.0),
                ),
                dim=-1,
            )
            statistics = statistics.masked_fill(~usable.unsqueeze(-1), 0.0)
        return statistics

    def forward(
        self,
        factors: Tensor,
        target_returns: Tensor,
        factor_mask: Tensor,
        feature_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """返回投影后的 `[B,D,E]` 证据及可审计的原始 `[B,D,6]` 统计。"""

        statistics = self.compute_statistics(
            factors,
            target_returns,
            factor_mask,
            feature_mask,
            target_mask,
        )
        evidence = self.projection(statistics)
        if output_dtype is not None:
            evidence = evidence.to(dtype=output_dtype)
        evidence = evidence.masked_fill(~factor_mask.unsqueeze(-1), 0.0)
        return evidence, statistics


class TargetAwareParentInteraction(nn.Module):
    """
    CausalRank 的目标感知父节点交互模块。

    输入：
        factor_features: [B, D, E] 或 [D, E]
        target_feature:  [B, E]    或 [E]
        factor_mask:     [B, D]    或 [D]，True 表示该 episode 中因子可观测
        factors:         [B,N,T,D] 或 [N,T,D]，用于计算条件增量证据
        target_returns:  [B,N,T]   或 [N,T]
        feature_mask:    与 factors 同形状的观测掩码
        target_mask:     与 target_returns 同形状的观测掩码

    输出：
        parent_features: [B, D, E] 或 [D, E]

    核心流程：

        H, h_Y
        ↓
        Target Relation Extraction
        ↓
        R^Y = (r_1^Y, ..., r_D^Y)
        ↓
        Explicit Conditional Incremental Evidence
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
        岭回归和偏相关是可微、正则化的条件证据近似，并不是对一般非线性
        SCM 的形式化条件独立检验。它们作为残差通道补充神经表示，最终的
        父节点判断仍由合成因果监督学习。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        conditional_ridge: float = 1e-2,
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

        if conditional_ridge <= 0.0:
            raise ValueError("conditional_ridge 必须为正数。")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.conditional_ridge = float(conditional_ridge)

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
        # 2. Explicit Conditional Incremental Evidence
        # ============================================================

        # 直接从原始 X/Y 面板计算控制全部其他候选后的岭回归与偏相关证据。
        # 它不使用标签，并通过残差连接补充而不是替代 target interaction。
        self.conditional_evidence_encoder = ConditionalIncrementalEvidenceEncoder(
            embedding_dim=embedding_dim,
            ridge=conditional_ridge,
        )
        self.conditional_evidence_norm = nn.LayerNorm(embedding_dim)

        # ============================================================
        # 3. Target Relevance Gate
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
        # 4. Target-Conditioned Conditional Competition
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
        # 5. Parent Evidence Fusion
        # ============================================================

        # 比较：
        #
        # r_j^Y
        # c_j
        # r_j^Y - c_j
        # r_j^Y * c_j
        # q_j^{cond}，显式条件增量证据
        #
        # 学习当前候选因子相对于其他 target-related
        # 候选变量的父节点证据。
        self.parent_representation_network = nn.Sequential(
            nn.LayerNorm(
                embedding_dim * 5
            ),
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

    def _validate_and_batch_panel_inputs(
        self,
        factors: Optional[Tensor],
        target_returns: Optional[Tensor],
        feature_mask: Optional[Tensor],
        target_mask: Optional[Tensor],
        batch_size: int,
        num_factors: int,
        remove_batch_dimension: bool,
    ) -> Tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        """校验显式条件证据所需的原始面板，并统一增加 batch 维。

        为兼容仅把预编码 H/h_Y 交给本模块的旧用法，X 和 Y 可以同时省略；
        但禁止只给其中一个，因为那会悄悄退化成含义不完整的条件统计。
        """

        if factors is None and target_returns is None:
            if feature_mask is not None or target_mask is not None:
                raise ValueError("未提供 factors/target_returns 时不能单独提供观测掩码。")
            return None, None, None, None
        if factors is None or target_returns is None:
            raise ValueError("factors 和 target_returns 必须同时提供或同时省略。")

        if remove_batch_dimension:
            if factors.ndim != 3 or target_returns.ndim != 2:
                raise ValueError(
                    "无 batch 表示输入对应的 factors/target_returns 必须为 "
                    "[N,T,D]/[N,T]。"
                )
            factors = factors.unsqueeze(0)
            target_returns = target_returns.unsqueeze(0)
            if feature_mask is not None:
                if feature_mask.ndim != 3:
                    raise ValueError("无 batch feature_mask 必须为 [N,T,D]。")
                feature_mask = feature_mask.unsqueeze(0)
            if target_mask is not None:
                if target_mask.ndim != 2:
                    raise ValueError("无 batch target_mask 必须为 [N,T]。")
                target_mask = target_mask.unsqueeze(0)
        elif factors.ndim != 4 or target_returns.ndim != 3:
            raise ValueError(
                "带 batch 表示输入对应的 factors/target_returns 必须为 "
                "[B,N,T,D]/[B,N,T]。"
            )

        if factors.shape[0] != batch_size or factors.shape[-1] != num_factors:
            raise ValueError("原始 factors 的 B、D 必须与 factor_features 对齐。")
        if target_returns.shape != factors.shape[:-1]:
            raise ValueError("target_returns 必须与 factors 的 [B,N,T] 对齐。")
        if feature_mask is not None and feature_mask.shape != factors.shape:
            raise ValueError("feature_mask 必须与 factors 形状相同。")
        if target_mask is not None and target_mask.shape != target_returns.shape:
            raise ValueError("target_mask 必须与 target_returns 形状相同。")
        return factors, target_returns, feature_mask, target_mask

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
        conditional_evidence: Optional[Tensor] = None,
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
            q_j^{cond}

        学习 parent-aware representation。
        """

        if conditional_evidence is None:
            # 保留旧的独立模块调用方式；没有原始面板时显式证据取零，而不是
            # 伪造某种条件关系。完整 CausalRankModel 始终会提供真实 X/Y。
            conditional_evidence = torch.zeros_like(factor_target_features)
        if conditional_evidence.shape != factor_target_features.shape:
            raise ValueError("conditional_evidence 必须与 factor_target_features 同形。")

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
                conditional_evidence,
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
        factors: Optional[Tensor] = None,
        target_returns: Optional[Tensor] = None,
        feature_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        完整计算：

            H, h_Y

            ↓

            Target Relation Extraction

            ↓

            R^Y

            ↓

            Explicit Ridge / Partial-Correlation Evidence

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
        batch_size, num_factors, _ = factor_features.shape
        (
            factors,
            target_returns,
            feature_mask,
            target_mask,
        ) = self._validate_and_batch_panel_inputs(
            factors,
            target_returns,
            feature_mask,
            target_mask,
            batch_size,
            num_factors,
            remove_batch_dimension,
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

        # Step 2：由原始面板计算显式条件增量。如果调用者只提供已经编码好的
        # H/h_Y，则该支路严格为零；完整训练模型会始终走有条件证据的路径。
        conditional_evidence = torch.zeros_like(factor_target_features)
        if factors is not None and target_returns is not None:
            conditional_evidence, _ = self.conditional_evidence_encoder(
                factors,
                target_returns,
                factor_mask,
                feature_mask,
                target_mask,
                output_dtype=factor_target_features.dtype,
            )
            # 在进入候选竞争前先注入条件证据，使 attention 的 query/key/value
            # 都能区分“边际相关但条件增量接近零”的代理变量。
            factor_target_features = self.conditional_evidence_norm(
                factor_target_features + conditional_evidence
            )
            factor_target_features = factor_target_features.masked_fill(
                ~factor_mask.unsqueeze(-1),
                0.0,
            )

        # Step 3
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

        # Step 4
        parent_features = self.fuse(
            factor_target_features,
            conditional_features,
            conditional_evidence,
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
