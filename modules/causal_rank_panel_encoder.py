"""CausalRank 的金融面板基础编码器。

本文件把 DAG-FM 的表格交互编码思想扩展到金融面板数据。模型先在每个
时间点联合观察所有股票的因子与对应收益，再结合市场状态沿时间轴汇总
历史信息，输出候选因子与目标收益各自的基础表示。

完整计算流程如下：

    X、Y -> 横截面 DAG-FM 编码 -> 加入市场状态 C -> 时间 Transformer
         -> 时间聚合 -> 候选因子基础表示 H 与目标基础表示 h_Y

输入张量约定：

    X: [批量 B, 股票 N, 时间 T, 因子 D]
    Y: [批量 B, 股票 N, 时间 T]
    C: [批量 B, 时间 T, 市场状态 Kc]
    asset_mask: [批量 B, 股票 N, 时间 T]
    feature_mask: [批量 B, 股票 N, 时间 T, 因子 D]
    target_mask: [批量 B, 股票 N, 时间 T]

输出张量约定：

    H:   [批量 B, 因子 D, 嵌入维度 E]
    h_Y: [批量 B, 嵌入维度 E]

编码器可从 X 与 Y 的联合横截面中学习一般统计关系，但不在末端显式执行
目标—因子交互或候选父节点之间的条件化。这两项职责由独立的
TargetAwareParentInteraction 完成。
"""

# 延迟解析类型标注，使较新的标注写法不会在导入阶段立即求值。
from __future__ import annotations

# argparse 用于保留一个可以从命令行直接运行的 main 测试入口。
import argparse
# sys 用于把相邻的 DAG-FM 目录临时加入 Python 模块搜索路径。
import sys
# Path 用于根据当前文件位置稳定定位 DAG-FM 编码器文件。
from pathlib import Path
# Optional 和 Tuple 用于标注可选输入及双张量输出。
from typing import Optional, Sequence, Tuple

# torch 提供张量运算、自动微分和随机测试数据。
import torch
# Tensor 是张量类型别名，nn 提供神经网络基础模块。
from torch import Tensor, nn
from torch.nn.parallel import data_parallel

# 计算当前文件所在的 modules 目录，避免依赖运行命令的当前工作目录。
_MODULE_DIRECTORY = Path(__file__).resolve().parent
# DAG-FM 与 modules 位于同一级，因此从父目录进入 DAG-FM 文件夹。
_DAGFM_DIRECTORY = _MODULE_DIRECTORY.parent / "DAG-FM"
# 把 Path 转成字符串，因为 sys.path 中保存的是字符串路径。
_DAGFM_DIRECTORY_STRING = str(_DAGFM_DIRECTORY)
# 只在路径尚未存在时插入，避免重复导入本文件后不断修改 sys.path。
if _DAGFM_DIRECTORY_STRING not in sys.path:
    # 把本地 DAG-FM 目录放到最前面，确保导入的是项目内的独立复现版本。
    sys.path.insert(0, _DAGFM_DIRECTORY_STRING)

# 复用已经实现的 DAG-FM 表格编码器。
from tabular_encoder import DAGFMTabularEncoder


class AttentionSequencePooling(nn.Module):
    """使用一个可学习查询向量把时间序列压缩成单个向量。

    普通平均池化给予每个时间点相同权重。这里的可学习查询向量会和每个
    时间点计算注意力分数，使模型能够根据训练任务自动决定哪些历史时期
    更值得保留。该模块不会改变批量维和嵌入维，只会移除时间维。
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        """构造时间注意力池化层。"""

        # 调用 nn.Module 初始化函数，使 PyTorch 可以登记后续参数和子模块。
        super().__init__()
        # 创建一个可训练查询向量，它可以理解为“我要从历史中寻找什么信息”。
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        # 创建多头注意力，让不同注意力头可以关注不同类型的时间模式。
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 层归一化用于稳定注意力输出的数值尺度。
        self.output_norm = nn.LayerNorm(embedding_dim)
        # Xavier 初始化让查询向量在训练开始时拥有合适的数值范围。
        nn.init.xavier_uniform_(self.query)

    def forward(
        self,
        sequence: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """将 `[批量, 时间, 嵌入]` 压缩为 `[批量, 嵌入]`。"""

        # 读取输入序列的批量大小，为每条时间序列共享同一个查询参数。
        batch_size = sequence.shape[0]
        # expand 创建查询向量的批量视图，不会复制出互相独立的训练参数。
        query = self.query.expand(batch_size, -1, -1)
        # 查询向量读取全部时间点；padding_mask 中 True 的时间点会被忽略。
        pooled, _ = self.attention(
            query=query,
            key=sequence,
            value=sequence,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        # 去掉长度为一的查询维，并通过层归一化得到固定长度时间摘要。
        return self.output_norm(pooled.squeeze(1))


class TargetFactorStatisticsEncoder(nn.Module):
    """把不会被深层集合池化抹去的 X-Y 直接统计证据编码为逐因子残差。

    合成目标包含线性、tanh、signed-square、sin 和 threshold 五类机制。
    这里计算前四类基函数与 Y 的有符号/绝对相关性，并附加有效观测比例。
    这些量只由模型可见的 X、Y 和 mask 得到，不读取 z、tau 或生成元数据。
    """

    num_statistics = 9

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(self.num_statistics),
            nn.Linear(self.num_statistics, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @staticmethod
    def _masked_correlation(
        transformed_factors: Tensor,
        centered_target: Tensor,
        mask: Tensor,
        count: Tensor,
    ) -> Tensor:
        transformed_mean = (
            transformed_factors * mask
        ).sum(dim=(1, 2)) / count
        centered_factors = (
            transformed_factors - transformed_mean[:, None, None, :]
        ) * mask
        covariance = (centered_factors * centered_target).sum(dim=(1, 2))
        factor_energy = centered_factors.square().sum(dim=(1, 2))
        target_energy = centered_target.square().sum(dim=(1, 2))
        denominator = (factor_energy * target_energy).clamp_min(1e-12).sqrt()
        return covariance / denominator

    def forward(
        self,
        factors: Tensor,
        targets: Tensor,
        feature_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """返回 `[B,D,E]` 的目标—因子统计残差。"""

        output_dtype = factors.dtype
        # 原始观测不需要梯度；固定统计量在 float32 中计算以避免 AMP 下
        # 小方差和相关系数分母发生下溢，投影网络本身仍然参与正常训练。
        with torch.no_grad():
            x = factors.float()
            y = targets.float().unsqueeze(-1)
            valid = (feature_mask & target_mask.unsqueeze(-1)).float()
            count = valid.sum(dim=(1, 2)).clamp_min(1.0)
            target_mean = (y * valid).sum(dim=(1, 2)) / count
            centered_target = (y - target_mean[:, None, None, :]) * valid

            correlations = []
            for transformed in (
                x,
                torch.tanh(2.0 * x),
                torch.sign(x) * x.square(),
                torch.sin(torch.pi * x),
            ):
                correlations.append(
                    self._masked_correlation(
                        transformed, centered_target, valid, count
                    )
                )
            signed = torch.stack(correlations, dim=-1)
            available_target = target_mask.sum(dim=(1, 2)).clamp_min(1)
            coverage = count / available_target[:, None].float()
            statistics = torch.cat((signed, signed.abs(), coverage.unsqueeze(-1)), dim=-1)

        return self.projection(statistics.to(dtype=output_dtype))


class CausalRankPanelEncoder(nn.Module):
    """把金融面板编码成候选因子和目标收益的基础统计表示。

    该类只负责横截面、市场状态与时间动态的联合编码。候选因子之间的
    条件关系以及因子相对于目标的交互由后续父节点感知模块负责。

    参数：
        market_state_dim: 市场状态 C 的特征数量；为 0 时表示不使用 C。
        embedding_dim: 所有隐向量的宽度。
        num_heads: 每个多头注意力层包含的注意力头数量。
        feedforward_dim: Transformer 内部前馈网络的隐藏层宽度。
        num_inducing_points: 横截面 ISAB 使用的可学习诱导点数量。
        num_seed_vectors: 横截面 PMA 为每个变量保留的摘要向量数量。
        num_cross_section_row_blocks: 横截面样本维 ISAB 数量。
        num_cross_section_column_blocks: 横截面变量维注意力块数量。
        num_temporal_layers: 沿时间轴运行的 Transformer 层数。
        max_time_steps: 学习型时间位置编码支持的最大历史窗口长度。
        dropout: 注意力和前馈网络使用的随机失活概率。
    """

    def __init__(
        self,
        market_state_dim: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        feedforward_dim: int = 256,
        num_inducing_points: int = 32,
        num_seed_vectors: int = 4,
        num_cross_section_row_blocks: int = 4,
        num_cross_section_column_blocks: int = 4,
        num_temporal_layers: int = 2,
        max_time_steps: int = 256,
        dropout: float = 0.0,
    ) -> None:
        """初始化横截面、市场状态与时间编码结构。"""

        # 调用 nn.Module 初始化函数，使 PyTorch 正确管理所有可训练参数。
        super().__init__()
        # 市场状态维度不能为负数，0 专门表示完全不使用市场状态输入。
        if market_state_dim < 0:
            raise ValueError("market_state_dim 不能为负数。")
        # 嵌入维度必须为正数，否则模型无法形成有效的隐藏表示。
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须为正整数。")
        # 注意力头数必须为正数，否则多头注意力无法工作。
        if num_heads <= 0:
            raise ValueError("num_heads 必须为正整数。")
        # 多头注意力会把嵌入维度平均分给每个头，因此二者必须整除。
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim 必须能够被 num_heads 整除。")
        # 时间位置编码至少要支持一个历史时间点。
        if max_time_steps <= 0:
            raise ValueError("max_time_steps 必须为正整数。")
        # 时间层允许设为零，以便后续进行移除时间交互的消融实验。
        if num_temporal_layers < 0:
            raise ValueError("时间层数不能为负数。")

        # 保存市场状态维度，供 forward 检查调用者是否正确提供 C。
        self.market_state_dim = market_state_dim
        # 保存统一嵌入维度，供 reshape 和输出形状构造使用。
        self.embedding_dim = embedding_dim
        # 保存最大时间长度，便于在输入过长时给出明确错误而不是隐式越界。
        self.max_time_steps = max_time_steps
        # 执行配置不属于模型参数：空元组表示单设备直接执行。
        self.cross_section_device_ids: Tuple[int, ...] = ()
        self.cross_section_chunk_size = 0
        self.cross_section_checkpoint = False

        # 横截面编码器在每个时间点联合读取 D 个因子和一个收益目标变量。
        self.cross_section_encoder = DAGFMTabularEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_inducing_points=num_inducing_points,
            num_seed_vectors=num_seed_vectors,
            num_row_blocks=num_cross_section_row_blocks,
            num_column_blocks=num_cross_section_column_blocks,
            dropout=dropout,
        )
        # 这个线性层为每个 PMA seed 产生一个标量重要性分数。
        self.seed_score = nn.Linear(embedding_dim, 1)
        # 因子角色向量告诉模型当前 Token 是普通候选因子。
        self.factor_role_embedding = nn.Parameter(torch.zeros(1, 1, 1, embedding_dim))
        # 收益角色向量告诉模型最后一个 Token 是需要研究的特殊目标 Y。
        self.target_role_embedding = nn.Parameter(torch.zeros(1, 1, 1, embedding_dim))
        nn.init.normal_(self.factor_role_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.target_role_embedding, mean=0.0, std=0.02)
        # 直接统计残差为每个因子保留与目标的低阶依赖信号，防止多层集合
        # 注意力在秩归一化边际分布上收敛到所有因子相同的表示。
        self.target_factor_statistics = TargetFactorStatisticsEncoder(embedding_dim)
        self.factor_statistics_norm = nn.LayerNorm(embedding_dim)

        # 只有 market_state_dim 大于零时才创建市场状态编码网络。
        if market_state_dim > 0:
            # MLP 把原始市场变量转换成与其他 Token 相同宽度的上下文向量。
            self.market_encoder: Optional[nn.Module] = nn.Sequential(
                nn.Linear(market_state_dim, feedforward_dim),
                nn.GELU(),
                nn.Linear(feedforward_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
        else:
            # 不使用市场状态时保存 None，从而不创建任何无用参数。
            self.market_encoder = None

        # 创建可学习时间位置编码，使模型能够区分较早和较近的历史观测。
        self.time_position_embedding = nn.Parameter(
            torch.empty(1, max_time_steps, embedding_dim)
        )
        # 用较小标准差初始化时间位置，避免其在训练初期压过数据本身的信息。
        nn.init.normal_(self.time_position_embedding, mean=0.0, std=0.02)

        # 只有时间层数大于零时才构造时间 Transformer。
        if num_temporal_layers > 0:
            # 单个时间 Transformer 层包含时间自注意力和逐时间点前馈网络。
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # 堆叠多个相同结构但参数独立的时间 Transformer 层。
            self.temporal_encoder: Optional[nn.Module] = nn.TransformerEncoder(
                encoder_layer=temporal_layer,
                num_layers=num_temporal_layers,
                norm=nn.LayerNorm(embedding_dim),
                enable_nested_tensor=False,
            )
        else:
            # 消融时间交互时保留 None，时间维仍会通过后面的注意力池化聚合。
            self.temporal_encoder = None

        # 使用可学习查询把长度可变的历史时间序列压缩为一个向量。
        self.temporal_pooling = AttentionSequencePooling(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def configure_cross_section_execution(
        self,
        device_ids: Sequence[int] = (),
        chunk_size: int = 0,
        use_checkpoint: bool = False,
    ) -> None:
        """配置横截面执行策略，不改变模型结构、参数或计算定义。

        多个 ``device_ids`` 会沿独立的 ``B*T`` 横截面表维并行计算；结果
        按原顺序汇总到第一个设备。checkpoint 保留随机数状态并在反向阶段
        重算横截面编码，用于进一步压低激活峰值。
        """

        normalized_ids = tuple(int(device_id) for device_id in device_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("cross-section device_ids 不能包含重复设备。")
        if any(device_id < 0 for device_id in normalized_ids):
            raise ValueError("cross-section device id 不能为负数。")
        if chunk_size < 0:
            raise ValueError("cross-section chunk_size 不能为负数。")
        self.cross_section_device_ids = normalized_ids
        self.cross_section_chunk_size = int(chunk_size)
        self.cross_section_checkpoint = bool(use_checkpoint)
        self.cross_section_encoder.set_activation_checkpoint(use_checkpoint)

    def _encode_cross_section_tables(
        self,
        tables: Tensor,
        observation_mask: Optional[Tensor],
        variable_embeddings: Tensor,
    ) -> Tensor:
        """编码一组彼此独立的横截面表，可沿表批量维分发到多张 GPU。"""

        if len(self.cross_section_device_ids) > 1:
            module_kwargs = (
                {
                    "observation_mask": observation_mask,
                    "variable_embeddings": variable_embeddings,
                }
            )
            return data_parallel(
                self.cross_section_encoder,
                (tables,),
                device_ids=self.cross_section_device_ids,
                output_device=self.cross_section_device_ids[0],
                module_kwargs=module_kwargs,
            )
        return self.cross_section_encoder(
            tables,
            observation_mask=observation_mask,
            variable_embeddings=variable_embeddings,
        )

    def _encode_cross_section_tables_checkpointed(
        self,
        tables: Tensor,
        observation_mask: Optional[Tensor],
        variable_embeddings: Tensor,
    ) -> Tensor:
        """编码横截面；激活重算由 DAG-FM replica 内部逐 block 执行。"""

        return self._encode_cross_section_tables(
            tables, observation_mask, variable_embeddings
        )

    def _validate_and_batch_inputs(
        self,
        factors: Tensor,
        target_returns: Tensor,
        market_state: Optional[Tensor],
        time_padding_mask: Optional[Tensor],
        asset_mask: Optional[Tensor],
        feature_mask: Optional[Tensor],
        target_mask: Optional[Tensor],
    ) -> Tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        bool,
    ]:
        """检查输入并把数值张量与三类观测掩码统一成批量形式。"""

        # 三维 X 表示调用者传入单个 [N,T,D] 面板，需要临时增加批量维。
        remove_batch_dimension = factors.ndim == 3
        # 只允许无批量的三维 X 或带批量的四维 X。
        if factors.ndim not in (3, 4):
            raise ValueError("factors 必须是 [N,T,D] 或 [B,N,T,D]。")
        # 无批量输入需要同步给全部张量增加相同的临时批量维。
        if remove_batch_dimension:
            # 把 X 从 [N,T,D] 转为 [1,N,T,D]。
            factors = factors.unsqueeze(0)
            # 无批量情况下 Y 必须是 [N,T]，随后转为 [1,N,T]。
            if target_returns.ndim != 2:
                raise ValueError("当 factors 为 [N,T,D] 时，target_returns 必须为 [N,T]。")
            # 增加 Y 的临时批量维。
            target_returns = target_returns.unsqueeze(0)
            # 如果提供 C，则无批量情况下 C 必须是 [T,Kc]。
            if market_state is not None:
                # 检查无批量市场状态的维度数量。
                if market_state.ndim != 2:
                    raise ValueError("无批量输入的 market_state 必须为 [T,Kc]。")
                # 把 C 从 [T,Kc] 转为 [1,T,Kc]。
                market_state = market_state.unsqueeze(0)
            # 如果提供时间掩码，则无批量情况下掩码必须是 [T]。
            if time_padding_mask is not None:
                # 检查无批量时间掩码的维度数量。
                if time_padding_mask.ndim != 1:
                    raise ValueError("无批量输入的 time_padding_mask 必须为 [T]。")
                # 把时间掩码从 [T] 转为 [1,T]。
                time_padding_mask = time_padding_mask.unsqueeze(0)
            # asset_mask 和 target_mask 都与 Y 一样采用 [N,T]。
            if asset_mask is not None:
                if asset_mask.ndim != 2:
                    raise ValueError("无批量输入的 asset_mask 必须为 [N,T]。")
                asset_mask = asset_mask.unsqueeze(0)
            if target_mask is not None:
                if target_mask.ndim != 2:
                    raise ValueError("无批量输入的 target_mask 必须为 [N,T]。")
                target_mask = target_mask.unsqueeze(0)
            # feature_mask 与 X 一样采用 [N,T,D]。
            if feature_mask is not None:
                if feature_mask.ndim != 3:
                    raise ValueError("无批量输入的 feature_mask 必须为 [N,T,D]。")
                feature_mask = feature_mask.unsqueeze(0)
        else:
            # 带批量 X 时，Y 必须同时带有 [B,N,T] 三个维度。
            if target_returns.ndim != 3:
                raise ValueError("当 factors 为 [B,N,T,D] 时，target_returns 必须为 [B,N,T]。")
            # 带批量情况下提供的 C 必须采用 [B,T,Kc] 形状。
            if market_state is not None and market_state.ndim != 3:
                raise ValueError("带批量输入的 market_state 必须为 [B,T,Kc]。")
            # 带批量情况下提供的时间掩码必须采用 [B,T] 形状。
            if time_padding_mask is not None and time_padding_mask.ndim != 2:
                raise ValueError("带批量输入的 time_padding_mask 必须为 [B,T]。")
            if asset_mask is not None and asset_mask.ndim != 3:
                raise ValueError("带批量输入的 asset_mask 必须为 [B,N,T]。")
            if target_mask is not None and target_mask.ndim != 3:
                raise ValueError("带批量输入的 target_mask 必须为 [B,N,T]。")
            if feature_mask is not None and feature_mask.ndim != 4:
                raise ValueError("带批量输入的 feature_mask 必须为 [B,N,T,D]。")

        # 从标准化后的 X 读取批量、股票、时间和候选因子数量。
        batch_size, num_assets, num_times, num_factors = factors.shape
        # 所有结构维度都必须非空，否则注意力层无法形成有效集合。
        if min(batch_size, num_assets, num_times, num_factors) <= 0:
            raise ValueError("批量、股票、时间和因子数量都必须大于零。")
        # Y 必须和 X 的批量、股票、时间三个维度逐一对应。
        if target_returns.shape != (batch_size, num_assets, num_times):
            raise ValueError("target_returns 的 [B,N,T] 必须与 factors 完全对应。")
        # 输入时间长度不能超过已经分配的时间位置编码长度。
        if num_times > self.max_time_steps:
            raise ValueError(
                f"输入时间长度 {num_times} 超过 max_time_steps={self.max_time_steps}。"
            )
        # X 和 Y 必须为浮点张量，因为线性层与注意力层不能直接处理整数张量。
        if not torch.is_floating_point(factors) or not torch.is_floating_point(target_returns):
            raise TypeError("factors 和 target_returns 都必须是浮点张量。")
        # 当前模块不在内部插补缺失值，因此 X 和 Y 中不能含 NaN 或 Inf。
        if not torch.isfinite(factors).all() or not torch.isfinite(target_returns).all():
            raise ValueError("factors 和 target_returns 中不能包含 NaN 或 Inf。")
        # X 与 Y 会直接拼接，因此必须位于同一设备并采用同一种浮点类型。
        if factors.device != target_returns.device:
            raise ValueError("factors 和 target_returns 必须位于同一设备。")
        if factors.dtype != target_returns.dtype:
            raise TypeError("factors 和 target_returns 必须使用相同 dtype。")

        # 未提供 asset_mask 时保持向后兼容，默认所有股票—时间位置都存在。
        expected_asset_shape = (batch_size, num_assets, num_times)
        if asset_mask is None:
            asset_mask = torch.ones(
                expected_asset_shape,
                dtype=torch.bool,
                device=factors.device,
            )
        else:
            if asset_mask.shape != expected_asset_shape:
                raise ValueError(
                    f"asset_mask 应为 {expected_asset_shape}，"
                    f"实际为 {tuple(asset_mask.shape)}。"
                )
            if asset_mask.dtype != torch.bool:
                raise TypeError("asset_mask 必须是布尔张量，True 表示股票月存在。")
            if asset_mask.device != factors.device:
                raise ValueError("asset_mask 和 factors 必须位于同一设备。")

        # feature_mask 缺省时，存在的股票月默认其全部 D 个因子都被观测。
        expected_feature_shape = (
            batch_size,
            num_assets,
            num_times,
            num_factors,
        )
        if feature_mask is None:
            feature_mask = asset_mask.unsqueeze(-1).expand(expected_feature_shape)
        else:
            if feature_mask.shape != expected_feature_shape:
                raise ValueError(
                    f"feature_mask 应为 {expected_feature_shape}，"
                    f"实际为 {tuple(feature_mask.shape)}。"
                )
            if feature_mask.dtype != torch.bool:
                raise TypeError("feature_mask 必须是布尔张量，True 表示因子已观测。")
            if feature_mask.device != factors.device:
                raise ValueError("feature_mask 和 factors 必须位于同一设备。")

        # target_mask 缺省时，Y 的有效范围默认等于股票月存在范围。
        if target_mask is None:
            target_mask = asset_mask
        else:
            if target_mask.shape != expected_asset_shape:
                raise ValueError(
                    f"target_mask 应为 {expected_asset_shape}，"
                    f"实际为 {tuple(target_mask.shape)}。"
                )
            if target_mask.dtype != torch.bool:
                raise TypeError("target_mask 必须是布尔张量，True 表示 Y 有效。")
            if target_mask.device != factors.device:
                raise ValueError("target_mask 和 factors 必须位于同一设备。")

        # 具体特征或 Y 只有在对应股票月存在时才可能有效。
        if (feature_mask & ~asset_mask.unsqueeze(-1)).any():
            raise ValueError("feature_mask 不能在 asset_mask=False 的位置为 True。")
        if (target_mask & ~asset_mask).any():
            raise ValueError("target_mask 不能在 asset_mask=False 的位置为 True。")

        # 声明使用市场状态的模型必须在每次前向传播时收到 C。
        if self.market_state_dim > 0 and market_state is None:
            raise ValueError("market_state_dim 大于零时必须提供 market_state。")
        # 声明不使用市场状态的模型不应收到多余的 C，以免调用含义不清。
        if self.market_state_dim == 0 and market_state is not None:
            raise ValueError("market_state_dim 为零时不应提供 market_state。")
        # 如果当前模型需要 C，则继续检查它的完整形状和数值类型。
        if market_state is not None:
            # C 的批量、时间和特征维必须与模型声明及 X 的形状一致。
            expected_market_shape = (batch_size, num_times, self.market_state_dim)
            # 实际形状不一致时立即报错，防止广播产生难以察觉的错误。
            if market_state.shape != expected_market_shape:
                raise ValueError(
                    f"market_state 应为 {expected_market_shape}，实际为 {tuple(market_state.shape)}。"
                )
            # 市场状态同样必须为浮点张量。
            if not torch.is_floating_point(market_state):
                raise TypeError("market_state 必须是浮点张量。")
            # 市场状态中也不能包含尚未处理的 NaN 或 Inf。
            if not torch.isfinite(market_state).all():
                raise ValueError("market_state 中不能包含 NaN 或 Inf。")
            # C 会逐元素加到面板表示上，因此其设备和 dtype 必须与 X 一致。
            if market_state.device != factors.device:
                raise ValueError("market_state 和 factors 必须位于同一设备。")
            if market_state.dtype != factors.dtype:
                raise TypeError("market_state 和 factors 必须使用相同 dtype。")

        # 若未提供时间掩码，则从股票存在性自动推导完全空缺的时间点。
        derived_time_padding_mask = ~asset_mask.any(dim=1)
        if time_padding_mask is None:
            time_padding_mask = derived_time_padding_mask
        else:
            # 掩码必须与每个批次的 T 个时间点逐一对应。
            if time_padding_mask.shape != (batch_size, num_times):
                raise ValueError("time_padding_mask 必须为 [B,T]。")
            # PyTorch 约定布尔值 True 表示该时间点属于填充并应被忽略。
            if time_padding_mask.dtype != torch.bool:
                raise TypeError("time_padding_mask 必须是布尔张量，True 表示忽略。")
            # 掩码必须和输入在同一设备，才能用于对应设备上的注意力运算。
            if time_padding_mask.device != factors.device:
                raise ValueError("time_padding_mask 和 factors 必须位于同一设备。")
            # 所有股票都不存在的月份必须被屏蔽；允许调用者额外屏蔽其他月份。
            if (derived_time_padding_mask & ~time_padding_mask).any():
                raise ValueError("所有股票均不存在的时间点必须在 time_padding_mask 中标记。")
        # 一条序列若全部被忽略，最终无法形成任何历史摘要。
        if time_padding_mask.all(dim=1).any():
            raise ValueError("每个面板至少需要一个未被掩码的有效时间点。")

        # 返回统一带批量维的输入以及最终是否需要移除批量维的标记。
        return (
            factors,
            target_returns,
            market_state,
            time_padding_mask,
            asset_mask,
            feature_mask,
            target_mask,
            remove_batch_dimension,
        )

    def encode_cross_section(
        self,
        factors: Tensor,
        target_returns: Tensor,
        joint_observation_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """在每个时间点联合编码所有股票的 X 和 Y。

        输入：
            factors: `[B,N,T,D]`。
            target_returns: `[B,N,T]`。
            joint_observation_mask: `[B,N,T,D+1]`，True 表示有效观测。

        输出：
            `[B,T,D+1,E]`，最后一个变量位置始终对应收益目标 Y。
        """

        # 解包形状，后续需要把 B 和 T 合并成独立横截面数据集的批量维。
        batch_size, num_assets, num_times, num_factors = factors.shape
        # 把时间维移到股票维之前，形成 [B,T,N,D] 的逐时间横截面布局。
        factors_by_time = factors.permute(0, 2, 1, 3)
        # 合并 B 和 T，使每个时间点成为一张形状为 [N,D] 的普通表格。
        factor_tables = factors_by_time.reshape(
            batch_size * num_times,
            num_assets,
            num_factors,
        )
        # 对 Y 做相同的时间优先变换，保持每只股票的因子和收益严格对齐。
        targets_by_time = target_returns.permute(0, 2, 1)
        # 为 Y 增加变量维，形成每张横截面表格中的最后一列。
        target_tables = targets_by_time.reshape(batch_size * num_times, num_assets, 1)
        # 将 D 个候选因子和一个目标收益拼成 [B*T,N,D+1] 的联合观测表。
        joint_tables = torch.cat((factor_tables, target_tables), dim=-1)
        # 观测掩码执行相同的时间优先变换，与联合表逐单元严格对应。
        joint_table_mask: Optional[Tensor] = None
        if joint_observation_mask is not None:
            joint_table_mask = joint_observation_mask.permute(0, 2, 1, 3).reshape(
                batch_size * num_times,
                num_assets,
                num_factors + 1,
            )
        # DAG-FM 编码器沿股票集合和变量集合提取联合统计表示。不同时间点
        # 在此阶段彼此独立，因此可以沿 B*T 精确分块或分发到多张 GPU。
        total_tables = joint_tables.shape[0]
        # 在 DAG-FM 的第一层之前标出最后一列是目标 Y。候选因子共用同一个
        # role，不引入因子编号，因此因子置换等变性保持不变。
        factor_roles = self.factor_role_embedding.reshape(
            1, 1, self.embedding_dim
        ).expand(total_tables, num_factors, -1)
        target_roles = self.target_role_embedding.reshape(
            1, 1, self.embedding_dim
        ).expand(total_tables, 1, -1)
        variable_embeddings = torch.cat((factor_roles, target_roles), dim=1)
        chunk_size = self.cross_section_chunk_size or total_tables
        seeded_chunks = []
        for start in range(0, total_tables, chunk_size):
            stop = min(start + chunk_size, total_tables)
            mask_chunk = (
                None
                if joint_table_mask is None
                else joint_table_mask[start:stop]
            )
            seeded_chunks.append(
                self._encode_cross_section_tables_checkpointed(
                    joint_tables[start:stop],
                    mask_chunk,
                    variable_embeddings[start:stop],
                )
            )
        seeded_features = (
            seeded_chunks[0]
            if len(seeded_chunks) == 1
            else torch.cat(seeded_chunks, dim=0)
        )
        # 为每个 seed 计算一个可学习标量分数，形状变为 [B*T,D+1,S]。
        seed_logits = self.seed_score(seeded_features).squeeze(-1)
        # softmax 把每组 seed 分数归一化成和为一的非负权重。
        seed_weights = torch.softmax(seed_logits, dim=-1)
        # 用注意力权重加权所有 seed，得到每个变量一个 E 维横截面摘要。
        cross_section_features = (
            seeded_features * seed_weights.unsqueeze(-1)
        ).sum(dim=-2)
        # 恢复独立的批量维和时间维，得到 [B,T,D+1,E]。
        cross_section_features = cross_section_features.reshape(
            batch_size,
            num_times,
            num_factors + 1,
            self.embedding_dim,
        )
        # 输入端已经注入角色；输出端保留同一角色残差，避免深层池化再次
        # 淡化目标身份，同时仍不编码具体候选因子的列编号。
        factor_features = (
            cross_section_features[:, :, :num_factors]
            + self.factor_role_embedding
        )
        # 给最后一个收益变量加入不同的目标身份提示，使模型明确知道研究对象是 Y。
        target_features = (
            cross_section_features[:, :, num_factors:]
            + self.target_role_embedding
        )
        # 重新拼接因子和目标，保持最后一个变量位置固定对应 Y。
        return torch.cat((factor_features, target_features), dim=2)

    def encode_market_state(self, market_state: Tensor) -> Tensor:
        """将 `[B,T,Kc]` 的市场变量编码成 `[B,T,E]`。"""

        # 调用该函数意味着初始化时已经声明使用市场状态，因此编码器不能为 None。
        if self.market_encoder is None:
            raise RuntimeError("当前模型没有启用市场状态编码器。")
        # 通过 MLP 对每个时间点独立编码，不在这里混合不同时间点的信息。
        return self.market_encoder(market_state)

    def encode_time(
        self,
        cross_section_features: Tensor,
        market_features: Optional[Tensor],
        time_padding_mask: Optional[Tensor],
        variable_time_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """沿历史时间轴编码并输出 `[B,D+1,E]` 的变量摘要。

        ``variable_time_padding_mask`` 为 `[B,D+1,T]`，用于排除某个变量在
        某个时间点没有任何有效横截面观测的情况。
        """

        # 从横截面表示中读取批量、时间和变量数量，其中变量数量等于 D+1。
        batch_size, num_times, num_variables, _ = cross_section_features.shape
        # 市场状态是每个时间点共享的上下文，因此扩展一个变量维后加到全部变量。
        if market_features is not None:
            cross_section_features = (
                cross_section_features + market_features.unsqueeze(2)
            )
        # 把变量维放到时间维之前，便于分别处理每个变量自己的历史序列。
        variable_sequences = cross_section_features.permute(0, 2, 1, 3)
        # 合并批量和变量维，让 Transformer 收到 [B*(D+1),T,E]。
        variable_sequences = variable_sequences.reshape(
            batch_size * num_variables,
            num_times,
            self.embedding_dim,
        )
        # 截取前 T 个位置编码，并把时间顺序信息加入每个变量的历史表示。
        variable_sequences = (
            variable_sequences + self.time_position_embedding[:, :num_times]
        )
        # 时间掩码需要匹配合并后的 [B*(D+1),T] 布局。
        expanded_time_mask: Optional[Tensor] = None
        # 优先使用逐变量掩码，它比全局时间掩码包含更精确的缺失信息。
        if variable_time_padding_mask is not None:
            if variable_time_padding_mask.shape != (
                batch_size,
                num_variables,
                num_times,
            ):
                raise ValueError("variable_time_padding_mask 必须为 [B,D+1,T]。")
            if variable_time_padding_mask.dtype != torch.bool:
                raise TypeError("variable_time_padding_mask 必须是布尔张量。")
            if variable_time_padding_mask.device != cross_section_features.device:
                raise ValueError(
                    "variable_time_padding_mask 和 cross_section_features "
                    "必须位于同一设备。"
                )
            expanded_time_mask = variable_time_padding_mask.reshape(
                batch_size * num_variables,
                num_times,
            )
        # 全局时间 padding 对一个面板内的全部变量共同生效。
        if time_padding_mask is not None:
            # 从 [B,T] 扩展为 [B,D+1,T]，同一面板的所有变量共享有效时间点。
            global_time_mask = time_padding_mask.unsqueeze(1).expand(
                batch_size,
                num_variables,
                num_times,
            ).reshape(
                batch_size * num_variables,
                num_times,
            )
            expanded_time_mask = (
                global_time_mask
                if expanded_time_mask is None
                else expanded_time_mask | global_time_mask
            )
        # 被屏蔽的时间表示在进入 Transformer 前归零，避免填充值参与残差。
        if expanded_time_mask is not None:
            variable_sequences = variable_sequences.masked_fill(
                expanded_time_mask.unsqueeze(-1),
                0.0,
            )
            # 完全没有历史观测的变量临时开放一个零 dummy，避免注意力产生 NaN。
            safe_time_mask = expanded_time_mask.clone()
            all_times_masked = safe_time_mask.all(dim=1)
            if all_times_masked.any():
                safe_time_mask[all_times_masked, 0] = False
            expanded_time_mask = safe_time_mask
        # 时间 Transformer 允许每个时间点读取同一历史窗口内的其他时间点。
        if self.temporal_encoder is not None:
            # src_key_padding_mask 中为 True 的填充时间点不会贡献注意力键和值。
            variable_sequences = self.temporal_encoder(
                variable_sequences,
                src_key_padding_mask=expanded_time_mask,
            )
        # 注意力池化根据训练任务对不同历史时期加权，移除时间维。
        pooled_variables = self.temporal_pooling(
            variable_sequences,
            padding_mask=expanded_time_mask,
        )
        # 恢复批量和变量维，返回 [B,D+1,E] 的历史变量摘要。
        return pooled_variables.reshape(
            batch_size,
            num_variables,
            self.embedding_dim,
        )

    def forward(
        self,
        factors: Tensor,
        target_returns: Tensor,
        market_state: Optional[Tensor] = None,
        time_padding_mask: Optional[Tensor] = None,
        asset_mask: Optional[Tensor] = None,
        feature_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """执行基础面板编码并返回 `(factor_features, target_feature)`。

        `time_padding_mask` 使用 PyTorch 约定：True 表示对应时间点是填充位置，
        应当被时间注意力和时间池化忽略；False 表示真实有效观测。
        其余三类 mask 使用数据生成协议：True 表示对应观测有效。

        批量输入返回 `[B,D,E]` 与 `[B,E]`；单面板输入返回 `[D,E]` 与 `[E]`。
        二者均未经过显式的因子条件上下文或目标—因子交互。
        """

        # 统一输入形状并提前阻止维度错配、整数输入和非有限值传播进模型。
        (
            factors,
            target_returns,
            market_state,
            time_padding_mask,
            asset_mask,
            feature_mask,
            target_mask,
            remove_batch_dimension,
        ) = self._validate_and_batch_inputs(
            factors,
            target_returns,
            market_state,
            time_padding_mask,
            asset_mask,
            feature_mask,
            target_mask,
        )
        # 将 D 个因子的观测状态和目标观测状态拼成 [B,N,T,D+1]。
        joint_observation_mask = torch.cat(
            (
                feature_mask,
                target_mask.unsqueeze(-1),
            ),
            dim=-1,
        )
        # 在每个时间点联合编码全部股票的因子和对应收益。
        cross_section_features = self.encode_cross_section(
            factors,
            target_returns,
            joint_observation_mask,
        )
        # 若某个变量在一个月份没有任何有效股票，则时间模块也应忽略该表示。
        variable_time_padding_mask = ~joint_observation_mask.any(dim=1)
        variable_time_padding_mask = variable_time_padding_mask.permute(0, 2, 1)
        # 默认不使用市场上下文，只有提供 C 时才计算市场状态表示。
        market_features: Optional[Tensor] = None
        # 启用市场状态的模型会把 C 转换成 [B,T,E] 的时间上下文。
        if market_state is not None:
            market_features = self.encode_market_state(market_state)
        # 沿时间轴编码每个因子和收益目标的历史变化并聚合时间维。
        temporal_features = self.encode_time(
            cross_section_features,
            market_features,
            time_padding_mask,
            variable_time_padding_mask,
        )
        # 前 D 个变量位置属于候选因子，最后一个位置属于目标收益 Y。
        factor_features = temporal_features[:, :-1]
        statistical_features = self.target_factor_statistics(
            factors,
            target_returns,
            feature_mask,
            target_mask,
        )
        factor_features = self.factor_statistics_norm(
            factor_features + statistical_features
        )
        # 提取最后一个目标位置，得到每个面板一个 E 维收益目标表示。
        target_feature = temporal_features[:, -1]
        # 若调用者输入单个无批量面板，则同步移除两个输出的临时批量维。
        if remove_batch_dimension:
            factor_features = factor_features.squeeze(0)
            target_feature = target_feature.squeeze(0)
        # 后续父节点感知模块接收 H 与 h_Y，并独占末端交互职责。
        return factor_features, target_feature


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造 main 自检入口的命令行参数。"""

    # 创建参数解析器，使研究者可以用小规模数据快速理解各维度含义。
    parser = argparse.ArgumentParser(description="测试 CausalRank 金融面板编码器")
    # B 表示一次送入模型的独立金融面板数量。
    parser.add_argument("--batch-size", type=int, default=2, help="批量面板数量 B")
    # N 表示每个横截面包含的股票数量。
    parser.add_argument("--num-assets", type=int, default=24, help="股票数量 N")
    # T 表示每个面板包含的历史时间点数量。
    parser.add_argument("--num-times", type=int, default=8, help="历史长度 T")
    # D 表示需要判断和排序的候选金融因子数量。
    parser.add_argument("--num-factors", type=int, default=6, help="候选因子数量 D")
    # Kc 表示每个时间点市场状态变量的数量。
    parser.add_argument("--market-dim", type=int, default=3, help="市场状态维度 Kc")
    # E 表示模型内部每个 Token 的隐藏向量宽度。
    parser.add_argument("--embedding-dim", type=int, default=32, help="嵌入维度 E")
    # 多头注意力头数必须能够整除嵌入维度。
    parser.add_argument("--num-heads", type=int, default=4, help="多头注意力头数")
    # 返回包含全部测试参数定义的解析器。
    return parser


def main() -> None:
    """运行双输出形状、置换性质和反向传播测试。"""

    # 读取命令行参数，允许在不修改源码的情况下改变测试张量规模。
    args = _build_argument_parser().parse_args()
    # 固定随机种子，使参数初始化、测试输入和随机置换可以重复。
    torch.manual_seed(42)
    # 创建适合快速 CPU 自检的小型面板编码器。
    model = CausalRankPanelEncoder(
        market_state_dim=args.market_dim,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        feedforward_dim=args.embedding_dim * 2,
        num_inducing_points=8,
        num_seed_vectors=2,
        num_cross_section_row_blocks=2,
        num_cross_section_column_blocks=2,
        num_temporal_layers=2,
        max_time_steps=max(32, args.num_times),
        dropout=0.0,
    )
    # 切换到评估模式，避免随机失活影响置换性质的数值比较。
    model.eval()

    # 随机生成 [B,N,T,D] 候选因子面板作为测试输入。
    factors = torch.randn(
        args.batch_size,
        args.num_assets,
        args.num_times,
        args.num_factors,
    )
    # 随机生成与股票和时间位置逐一对应的 [B,N,T] 收益观测。
    target_returns = torch.randn(
        args.batch_size,
        args.num_assets,
        args.num_times,
    )
    # Kc=0 表示不使用市场状态，否则生成每个时间点共享的 [B,T,Kc] 状态。
    market_state: Optional[Tensor]
    if args.market_dim == 0:
        market_state = None
    else:
        market_state = torch.randn(
            args.batch_size,
            args.num_times,
            args.market_dim,
        )
    # 生成与训练数据协议一致的三类 True-valid 掩码。
    asset_mask = torch.ones(
        args.batch_size,
        args.num_assets,
        args.num_times,
        dtype=torch.bool,
    )
    # 最后一只股票模拟横截面 padding；最后一个月模拟整个时间点 padding。
    if args.num_assets > 1:
        asset_mask[:, -1, :] = False
    if args.num_times > 1:
        asset_mask[0, :, -1] = False
    feature_mask = asset_mask.unsqueeze(-1).expand_as(factors).clone()
    target_mask = asset_mask.clone()
    # 制造变量级缺失，包括某个因子在一个月份完全无观测的情况。
    feature_mask[:, 0, 0, 0] = False
    if args.num_times > 1:
        feature_mask[:, :, 0, 0] = False
    target_mask[:, 0, 0] = False
    # time_padding_mask 使用 PyTorch 语义：True 表示整个时间点应忽略。
    time_padding_mask = ~asset_mask.any(dim=1)
    # 编码器固定输出候选因子基础表示 H 和目标基础表示 h_Y。
    factor_features, target_feature = model(
        factors,
        target_returns,
        market_state,
        time_padding_mask=time_padding_mask,
        asset_mask=asset_mask,
        feature_mask=feature_mask,
        target_mask=target_mask,
    )
    # 构造两个返回值各自的理论形状。
    expected_factor_shape = (
        args.batch_size,
        args.num_factors,
        args.embedding_dim,
    )
    expected_target_shape = (args.batch_size, args.embedding_dim)
    assert factor_features.shape == expected_factor_shape, "因子输出形状错误。"
    assert target_feature.shape == expected_target_shape, "目标输出形状错误。"

    # 使用第一张面板测试没有批量维的便捷调用接口。
    single_market_state = None if market_state is None else market_state[0]
    factor_single, target_single = model(
        factors[0],
        target_returns[0],
        single_market_state,
        time_padding_mask=time_padding_mask[0],
        asset_mask=asset_mask[0],
        feature_mask=feature_mask[0],
        target_mask=target_mask[0],
    )
    assert factor_single.shape == expected_factor_shape[1:], "单面板因子形状错误。"
    assert target_single.shape == expected_target_shape[1:], "单面板目标形状错误。"
    assert torch.allclose(factor_single, factor_features[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(target_single, target_feature[0], atol=1e-5, rtol=1e-5)

    # 构造一个股票排列，用于验证股票横截面被正确视为无序集合。
    asset_permutation = torch.randperm(args.num_assets)
    # X 和 Y 必须做相同股票置换，才能保持每只股票的因子收益配对。
    factor_assets_permuted, target_assets_permuted = model(
        factors[:, asset_permutation],
        target_returns[:, asset_permutation],
        market_state,
        time_padding_mask=time_padding_mask,
        asset_mask=asset_mask[:, asset_permutation],
        feature_mask=feature_mask[:, asset_permutation],
        target_mask=target_mask[:, asset_permutation],
    )
    # 股票是无序集合，因此同时换序 X、Y 不应改变 H 或 h_Y。
    asset_difference = max(
        (factor_features - factor_assets_permuted).abs().max().item(),
        (target_feature - target_assets_permuted).abs().max().item(),
    )

    # 构造一个因子排列，用于验证模型没有依赖固定的因子列编号。
    factor_permutation = torch.randperm(args.num_factors)
    # 只对 X 的因子维换序，收益目标仍固定在联合表格最后一列。
    factor_features_permuted, target_feature_permuted = model(
        factors[:, :, :, factor_permutation],
        target_returns,
        market_state,
        time_padding_mask=time_padding_mask,
        asset_mask=asset_mask,
        feature_mask=feature_mask[:, :, :, factor_permutation],
        target_mask=target_mask,
    )
    # H 应随因子输入等变；h_Y 作为目标摘要应对因子顺序保持不变。
    factor_difference = (
        factor_features[:, factor_permutation] - factor_features_permuted
    ).abs().max().item()
    target_difference = (
        target_feature - target_feature_permuted
    ).abs().max().item()

    # 改写所有缺失位置的有限填充值，结果必须保持不变。
    differently_filled_factors = factors.clone()
    differently_filled_targets = target_returns.clone()
    differently_filled_factors[~feature_mask] = 999.0
    differently_filled_targets[~target_mask] = -999.0
    filled_factor_features, filled_target_feature = model(
        differently_filled_factors,
        differently_filled_targets,
        market_state,
        time_padding_mask=time_padding_mask,
        asset_mask=asset_mask,
        feature_mask=feature_mask,
        target_mask=target_mask,
    )
    masked_fill_difference = max(
        (factor_features - filled_factor_features).abs().max().item(),
        (target_feature - filled_target_feature).abs().max().item(),
    )

    # 同时使用 H 与 h_Y 构造损失，确认两条输出路径均能接受下游监督。
    factor_weights = torch.linspace(
        0.5,
        1.5,
        args.embedding_dim,
        device=factor_features.device,
    )
    loss = (
        (factor_features * factor_weights).mean()
        + (target_feature * factor_weights).mean()
    )
    loss.backward()
    # 基础编码器中不应再存在只服务于旧末端交互路径的悬空参数。
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    gradient_sum = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    # 浮点注意力在改变计算顺序后可能存在微小舍入误差，因此允许 1e-5。
    tolerance = 1e-5
    # 股票置换不应实质改变编码结果。
    assert asset_difference < tolerance, "股票置换不变性测试失败。"
    # 因子置换应只改变输出顺序，不应改变各因子的表示内容。
    assert factor_difference < tolerance, "因子置换等变性测试失败。"
    assert target_difference < tolerance, "目标表示的因子置换不变性测试失败。"
    assert masked_fill_difference < tolerance, "缺失填充值泄漏到面板表示。"
    assert torch.isfinite(factor_features).all(), "因子表示包含 NaN 或 Inf。"
    assert torch.isfinite(target_feature).all(), "目标表示包含 NaN 或 Inf。"
    assert not missing_gradients, f"存在未参与前向传播的参数：{missing_gradients}"
    assert gradient_sum > 0.0, "反向传播测试失败。"

    # 统计所有可训练参数数量，帮助研究者了解当前测试模型的规模。
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    # 输出总体成功信息，表示所有断言均已通过。
    print("CausalRankPanelEncoder 测试通过")
    # 打印三个输入张量的形状，方便核对 B、N、T、D 和 Kc 的位置。
    print(f"X 输入形状：{tuple(factors.shape)}")
    print(f"Y 输入形状：{tuple(target_returns.shape)}")
    market_shape = None if market_state is None else tuple(market_state.shape)
    print(f"C 输入形状：{market_shape}")
    # H 与 h_Y 将直接交给 TargetAwareParentInteraction。
    print(f"因子基础表示 H：{tuple(factor_features.shape)}")
    print(f"目标基础表示 h_Y：{tuple(target_feature.shape)}")
    # 打印当前测试配置下的模型参数总量。
    print(f"参数数量：{parameter_count:,}")
    # 打印股票换序后的最大误差，数值应接近浮点计算精度。
    print(f"股票置换最大误差：{asset_difference:.3e}")
    # 打印因子换序后的最大误差，数值应接近浮点计算精度。
    print(f"因子置换最大误差：{factor_difference:.3e}")
    print(f"目标置换最大误差：{target_difference:.3e}")
    print(f"缺失填充值最大误差：{masked_fill_difference:.3e}")
    # 打印累计梯度大小，正数说明完整网络可以参与端到端训练。
    print(f"全部参数梯度绝对值之和：{gradient_sum:.3e}")


# 只有直接运行本文件时才执行 main；作为模块导入时不会自动运行测试。
if __name__ == "__main__":
    # 调用轻量级自检入口。
    main()
