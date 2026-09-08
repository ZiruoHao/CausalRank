"""DAG-FM 风格的表格交互编码器。

该文件根据 DAG-FM 论文中描述的 Tabular Interaction Block 独立复现，
不依赖尚未公开的官方实现。整体计算顺序为：

    标量嵌入 -> 样本维 ISAB -> 变量维 MAB -> 样本维 PMA 聚合

输入表格的形状为 ``[批量大小, 样本数, 变量数]``，可选观测掩码使用相同
形状且 True 表示有效单元。输出形状为
``[批量大小, 变量数, 种子数, 嵌入维度]``。输出中的每个变量都拥有
``种子数`` 个表示向量，可供后续叶节点判断器和父节点判断器使用。

设计上不添加样本或变量位置编码，因此：

1. 打乱样本顺序不会改变最终的变量表示（样本置换不变性）；
2. 打乱变量顺序会使输出发生相同的置换（变量置换等变性）。

参考：
    DAG-FM: A Foundation Model for Causal Discovery under
    Heterogeneous Causal Mechanisms, Section 4.4.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class MultiheadAttentionBlock(nn.Module):
    """Set Transformer 中的多头注意力块（MAB）。

    ``query`` 表示需要更新的一组元素，``key_value`` 表示它可以读取的
    上下文集合。当两者相同时，该模块就是集合上的自注意力；当两者不同时，
    它就是集合之间的交叉注意力。

    参数：
        embedding_dim: 每个元素的表示维度。
        num_heads: 多头注意力的头数，必须整除 ``embedding_dim``。
        feedforward_dim: 前馈网络的隐藏层维度。
        dropout: 注意力和前馈网络中的随机失活率。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} 必须能被 num_heads={num_heads} 整除。"
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """更新查询集合。

        参数：
            query: ``[批量, 查询元素数, 嵌入维度]``。
            key_value: ``[批量, 上下文元素数, 嵌入维度]``。
            key_padding_mask: 可选的 ``[批量, 上下文元素数]`` 布尔掩码；
                ``True`` 表示该位置不参与注意力。
        """

        attention_output, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(query + self.dropout(attention_output))
        output = self.feedforward_norm(hidden + self.dropout(self.feedforward(hidden)))
        return output


class InducedSetAttentionBlock(nn.Module):
    """诱导集合注意力块（ISAB）。

    直接在 ``n`` 个样本之间做自注意力的复杂度约为 ``O(n^2)``。ISAB 使用
    少量可学习的诱导点作为信息中介，把复杂度降低到约 ``O(nm)``，其中
    ``m`` 是诱导点数量：

        样本集合 -> 诱导点汇总信息 -> 信息返回样本集合。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_inducing_points: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_inducing_points <= 0:
            raise ValueError("num_inducing_points 必须为正整数。")

        self.inducing_points = nn.Parameter(
            torch.empty(1, num_inducing_points, embedding_dim)
        )
        self.inducing_to_input = MultiheadAttentionBlock(
            embedding_dim, num_heads, feedforward_dim, dropout
        )
        self.input_to_inducing = MultiheadAttentionBlock(
            embedding_dim, num_heads, feedforward_dim, dropout
        )
        nn.init.xavier_uniform_(self.inducing_points)

    def forward(
        self,
        inputs: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """对一个集合执行诱导注意力。

        参数：
            inputs: ``[批量, 集合大小, 嵌入维度]``。
            key_padding_mask: 可选的集合元素掩码。
        """

        batch_size = inputs.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)

        # 诱导点首先读取整个输入集合。
        inducing_hidden = self.inducing_to_input(
            inducing, inputs, key_padding_mask=key_padding_mask
        )
        # 每个输入元素再从诱导点中读取压缩后的全局信息。
        return self.input_to_inducing(inputs, inducing_hidden)


class PoolingByMultiheadAttention(nn.Module):
    """使用多头注意力进行池化（PMA）。

    与平均池化不同，PMA 使用 ``num_seed_vectors`` 个可学习查询向量从样本
    集合中提取信息，因此能够为每个变量保留多个不同侧面的统计表示。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_seed_vectors: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_seed_vectors <= 0:
            raise ValueError("num_seed_vectors 必须为正整数。")

        self.seed_vectors = nn.Parameter(
            torch.empty(1, num_seed_vectors, embedding_dim)
        )
        self.attention = MultiheadAttentionBlock(
            embedding_dim, num_heads, feedforward_dim, dropout
        )
        nn.init.xavier_uniform_(self.seed_vectors)

    def forward(
        self,
        inputs: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = inputs.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        return self.attention(seeds, inputs, key_padding_mask=key_padding_mask)


class DAGFMTabularEncoder(nn.Module):
    """DAG-FM 风格的表格统计特征提取器。

    参数：
        embedding_dim: 单元格嵌入和输出表示的维度。
        num_heads: 多头注意力头数。
        feedforward_dim: 每个注意力块中前馈网络的隐藏维度。
        num_inducing_points: 样本维 ISAB 使用的诱导点数量。
        num_seed_vectors: PMA 为每个变量产生的表示向量数量，即论文中的 k。
        num_row_blocks: 样本维 ISAB 数量；论文描述为 4。
        num_column_blocks: 变量维 MAB 数量；论文描述为 4。
        dropout: 随机失活率。

    输入：
        ``[样本数, 变量数]`` 或 ``[批量大小, 样本数, 变量数]``。

    输出：
        若输入带批量维，输出为
        ``[批量大小, 变量数, num_seed_vectors, embedding_dim]``；
        若输入不带批量维，则自动去掉输出的批量维。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        feedforward_dim: int = 256,
        num_inducing_points: int = 32,
        num_seed_vectors: int = 4,
        num_row_blocks: int = 4,
        num_column_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_row_blocks <= 0 or num_column_blocks <= 0:
            raise ValueError("行、列注意力块数量必须为正整数。")

        self.embedding_dim = embedding_dim
        self.num_seed_vectors = num_seed_vectors
        # 该开关只改变激活保存策略，不改变模型参数和前向计算定义。
        self.activation_checkpoint = False

        # 每个标量观测值独立映射到 embedding_dim 维。所有变量共享该映射，
        # 从而不会把某个固定列号编码进模型，保证变量置换等变性。
        self.scalar_embedding = nn.Linear(1, embedding_dim)

        # 对每个变量分别建模其跨样本分布。不同变量共享相同的 ISAB 参数。
        self.row_blocks = nn.ModuleList(
            [
                InducedSetAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    num_inducing_points=num_inducing_points,
                    dropout=dropout,
                )
                for _ in range(num_row_blocks)
            ]
        )

        # 在每个样本内部让所有变量相互交换信息，学习变量之间的统计依赖。
        # 不使用变量位置编码，使同一个模型能够处理不同的变量排列。
        self.column_blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_column_blocks)
            ]
        )

        # 最后把任意数量的样本聚合为固定数量的种子向量。
        self.sample_pooling = PoolingByMultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_seed_vectors=num_seed_vectors,
            dropout=dropout,
        )

    def set_activation_checkpoint(self, enabled: bool) -> None:
        """启用逐 block 激活重算；在 DataParallel replica 内部同样生效。"""

        self.activation_checkpoint = bool(enabled)

    def _checkpoint_enabled(self) -> bool:
        return (
            self.activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        )

    @staticmethod
    def _run_row_blocks(
        hidden: Tensor,
        blocks: tuple[nn.Module, ...],
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        for block in blocks:
            hidden = block(hidden, key_padding_mask=padding_mask)
        return hidden

    @staticmethod
    def _run_column_blocks(
        hidden: Tensor,
        blocks: tuple[nn.Module, ...],
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        for block in blocks:
            hidden = block(hidden, hidden, key_padding_mask=padding_mask)
        return hidden

    @staticmethod
    def _make_safe_padding_mask(padding_mask: Tensor) -> Tensor:
        """避免整行均被屏蔽时注意力 softmax 产生 NaN。

        全掩码集合没有可供注意力读取的键。这里仅临时开放第一个、已经被
        归零的 dummy 位置，使该集合形成有限的“无观测”表示，而不会把任意
        缺失填充值误认为真实观测。
        """

        # clone 防止修改调用者持有的原始掩码。
        safe_mask = padding_mask.clone()
        # 找到没有任何有效键的集合。
        all_masked = safe_mask.all(dim=1)
        # 临时开放一个有限的零值键，保证 softmax 有定义。
        if all_masked.any():
            safe_mask[all_masked, 0] = False
        return safe_mask

    def forward(
        self,
        table: Tensor,
        observation_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """将观测表格编码为逐变量表示。

        ``observation_mask`` 与 table 同形状，True 表示真实观测，False
        表示缺失或 padding。未提供时保持原有行为，即所有单元均有效。
        """

        if table.ndim not in (2, 3):
            raise ValueError(
                "table 必须是 [样本数, 变量数] 或 "
                f"[批量大小, 样本数, 变量数]，实际形状为 {tuple(table.shape)}。"
            )
        if not torch.is_floating_point(table):
            raise TypeError("table 必须是浮点张量；请先将整数数据转换为浮点数。")

        remove_batch_dimension = table.ndim == 2
        if remove_batch_dimension:
            table = table.unsqueeze(0)
            if observation_mask is not None:
                if observation_mask.ndim != 2:
                    raise ValueError("二维 table 的 observation_mask 也必须是二维。")
                observation_mask = observation_mask.unsqueeze(0)
        elif observation_mask is not None and observation_mask.ndim != 3:
            raise ValueError("三维 table 的 observation_mask 也必须是三维。")

        batch_size, num_samples, num_variables = table.shape
        if num_samples == 0 or num_variables == 0:
            raise ValueError("样本数和变量数都必须大于 0。")
        if not torch.isfinite(table).all():
            raise ValueError("输入包含 NaN 或 Inf；请在编码前完成缺失值处理。")

        # 检查并标准化观测掩码；True 表示有效，与 PyTorch padding mask 相反。
        if observation_mask is not None:
            if observation_mask.shape != table.shape:
                raise ValueError("observation_mask 必须与 table 具有完全相同的形状。")
            if observation_mask.dtype != torch.bool:
                raise TypeError("observation_mask 必须是布尔张量，True 表示有效。")
            if observation_mask.device != table.device:
                raise ValueError("observation_mask 和 table 必须位于同一设备。")
            # 无效值在嵌入前统一归零，使其具体填充值不可能泄漏到表示中。
            table = table.masked_fill(~observation_mask, 0.0)

        # [B, N, D] -> [B, N, D, E]
        hidden = self.scalar_embedding(table.unsqueeze(-1))

        # 样本维交互：把每个变量视为一个独立的样本集合。
        # [B, N, D, E] -> [B*D, N, E]
        hidden = hidden.permute(0, 2, 1, 3).reshape(
            batch_size * num_variables, num_samples, self.embedding_dim
        )
        row_padding_mask: Optional[Tensor] = None
        safe_row_padding_mask: Optional[Tensor] = None
        if observation_mask is not None:
            # 每个变量分别屏蔽未观测股票，形状为 [B*D,N]。
            row_padding_mask = ~observation_mask.permute(0, 2, 1).reshape(
                batch_size * num_variables,
                num_samples,
            )
            safe_row_padding_mask = self._make_safe_padding_mask(row_padding_mask)
        if self._checkpoint_enabled():
            # 两层为一组可少保存约一半层边界，同时避免整段重算时出现过高峰值。
            for start in range(0, len(self.row_blocks), 2):
                block_group = tuple(self.row_blocks[start : start + 2])
                hidden = checkpoint(
                    lambda values, current_blocks=block_group: self._run_row_blocks(
                        values,
                        current_blocks,
                        safe_row_padding_mask,
                    ),
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
        else:
            for block in self.row_blocks:
                hidden = block(hidden, key_padding_mask=safe_row_padding_mask)

        # 恢复表格布局，并在每个样本内部执行变量维交互。
        # [B*D, N, E] -> [B*N, D, E]
        hidden = hidden.reshape(
            batch_size, num_variables, num_samples, self.embedding_dim
        )
        hidden = hidden.permute(0, 2, 1, 3).reshape(
            batch_size * num_samples, num_variables, self.embedding_dim
        )
        column_padding_mask: Optional[Tensor] = None
        if observation_mask is not None:
            # 每个样本内部，缺失变量不能作为其他变量的 Key/Value。
            column_padding_mask = ~observation_mask.reshape(
                batch_size * num_samples,
                num_variables,
            )
            column_padding_mask = self._make_safe_padding_mask(
                column_padding_mask
            )
        if self._checkpoint_enabled():
            for start in range(0, len(self.column_blocks), 2):
                block_group = tuple(self.column_blocks[start : start + 2])
                hidden = checkpoint(
                    lambda values, current_blocks=block_group: self._run_column_blocks(
                        values,
                        current_blocks,
                        column_padding_mask,
                    ),
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
        else:
            for block in self.column_blocks:
                hidden = block(
                    hidden,
                    hidden,
                    key_padding_mask=column_padding_mask,
                )

        # PMA 在样本维聚合信息，为每个变量产生 K 个表示向量。
        # [B*N, D, E] -> [B*D, N, E] -> [B*D, K, E]
        hidden = hidden.reshape(
            batch_size, num_samples, num_variables, self.embedding_dim
        )
        hidden = hidden.permute(0, 2, 1, 3).reshape(
            batch_size * num_variables, num_samples, self.embedding_dim
        )
        # 一个变量若在整个横截面完全没有观测，其列交互表示可能读取到其他
        # 变量。PMA 又必须临时开放一个 dummy 键；若直接使用该表示，结果会
        # 依赖“哪个样本恰好排在第一位”。这里把这种全缺失变量的全部样本
        # 表示固定为零，再由 seed query 生成稳定的无观测摘要。
        if row_padding_mask is not None:
            all_samples_masked = row_padding_mask.all(dim=1)
            if all_samples_masked.any():
                hidden = hidden.masked_fill(
                    all_samples_masked[:, None, None],
                    0.0,
                )
        # PMA 必须再次使用样本维掩码，避免缺失 query 的中间表示进入聚合结果。
        if self._checkpoint_enabled():
            encoded = checkpoint(
                lambda values: self.sample_pooling(
                    values,
                    key_padding_mask=safe_row_padding_mask,
                ),
                hidden,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            encoded = self.sample_pooling(
                hidden,
                key_padding_mask=safe_row_padding_mask,
            )
        encoded = encoded.reshape(
            batch_size,
            num_variables,
            self.num_seed_vectors,
            self.embedding_dim,
        )

        if remove_batch_dimension:
            encoded = encoded.squeeze(0)
        return encoded


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造测试入口的命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="测试 DAG-FM 表格交互编码器")
    parser.add_argument("--batch-size", type=int, default=2, help="批量大小")
    parser.add_argument("--num-samples", type=int, default=64, help="每张表的样本数")
    parser.add_argument("--num-variables", type=int, default=10, help="变量数")
    parser.add_argument("--embedding-dim", type=int, default=64, help="嵌入维度")
    parser.add_argument("--num-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num-seeds", type=int, default=4, help="PMA 种子向量数")
    return parser


def main() -> None:
    """运行轻量级自检，保留为本文件的直接测试入口。"""

    args = _build_argument_parser().parse_args()
    torch.manual_seed(42)

    # 测试使用 dropout=0 并切换到 eval，方便严格检查置换性质。
    model = DAGFMTabularEncoder(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        feedforward_dim=args.embedding_dim * 2,
        num_inducing_points=16,
        num_seed_vectors=args.num_seeds,
        num_row_blocks=4,
        num_column_blocks=4,
        dropout=0.0,
    )
    model.eval()

    table = torch.randn(
        args.batch_size,
        args.num_samples,
        args.num_variables,
        requires_grad=True,
    )
    # True 表示真实观测；既制造普通缺失，也覆盖整行、整列无观测的边界情况。
    observation_mask = torch.rand(table.shape) > 0.20
    observation_mask[:, 0, 0] = False
    if args.num_samples > 1:
        observation_mask[:, -1, :] = False
        observation_mask[:, 0, 0] = True
    if args.num_variables > 1:
        observation_mask[0, :, -1] = False
    encoded = model(table, observation_mask=observation_mask)
    expected_shape = (
        args.batch_size,
        args.num_variables,
        args.num_seeds,
        args.embedding_dim,
    )
    assert encoded.shape == expected_shape

    # 样本顺序不应影响最终结果，因为行方向使用集合注意力且最终通过 PMA 聚合。
    sample_permutation = torch.randperm(args.num_samples)
    encoded_sample_permuted = model(
        table[:, sample_permutation, :],
        observation_mask=observation_mask[:, sample_permutation, :],
    )
    sample_difference = (encoded - encoded_sample_permuted).abs().max().item()

    # 变量顺序变化后，逐变量输出应发生相同的顺序变化。
    variable_permutation = torch.randperm(args.num_variables)
    encoded_variable_permuted = model(
        table[:, :, variable_permutation],
        observation_mask=observation_mask[:, :, variable_permutation],
    )
    variable_difference = (
        encoded[:, variable_permutation] - encoded_variable_permuted
    ).abs().max().item()

    # 掩码位置的具体有限填充值不应影响表示，Dataset 默认使用 0 只是存储约定。
    differently_filled_table = table.detach().clone()
    differently_filled_table[~observation_mask] = 999.0
    encoded_with_different_fill = model(
        differently_filled_table,
        observation_mask=observation_mask,
    )
    masked_fill_difference = (
        encoded.detach() - encoded_with_different_fill
    ).abs().max().item()

    # 确认计算图可用于正常训练。
    loss = encoded.square().mean()
    loss.backward()
    gradient_norm = model.scalar_embedding.weight.grad.norm().item()
    masked_input_gradient = table.grad[~observation_mask].abs().max().item()

    tolerance = 1e-5
    assert sample_difference < tolerance, "样本置换不变性测试失败。"
    assert variable_difference < tolerance, "变量置换等变性测试失败。"
    assert masked_fill_difference < tolerance, "缺失填充值泄漏到编码结果。"
    assert masked_input_gradient == 0.0, "缺失位置不应收到输入梯度。"
    assert torch.isfinite(encoded).all(), "全掩码边界产生了 NaN 或 Inf。"
    assert gradient_norm > 0.0, "反向传播测试失败。"

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("DAG-FM 表格编码器测试通过")
    print(f"输入形状：{tuple(table.shape)}")
    print(f"输出形状：{tuple(encoded.shape)}")
    print(f"参数数量：{parameter_count:,}")
    print(f"样本置换最大误差：{sample_difference:.3e}")
    print(f"变量置换最大误差：{variable_difference:.3e}")
    print(f"缺失填充值最大误差：{masked_fill_difference:.3e}")
    print(f"缺失位置输入梯度：{masked_input_gradient:.3e}")
    print(f"嵌入层梯度范数：{gradient_norm:.3e}")


if __name__ == "__main__":
    main()
