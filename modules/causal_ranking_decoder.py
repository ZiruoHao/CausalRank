"""CausalRank 的因果排序解码器。

该模块位于 TargetAwareParentInteraction 之后：

    parent_features E
        -> shared evidence decoder
        -> causal existence head
        -> non-negative effect-strength head
        -> soft causal gate
        -> ranking scores s

输入 parent_features 为 [B,D,E]，默认输出 [B,D] 的最终排序分数；同时支持
无 batch 的 [D,E] 输入并返回 [D]。训练时可通过 return_auxiliary=True
同时取得 existence logits、存在概率和效应强度，以分别构造分类、回归和
排序损失。

Decoder 不负责候选因子的填充、窗口采样或递归搜索。调用者应先构造一个
紧凑的候选因子集合，再把它交给完整模型。这样训练和推理共享完全相同的
解码逻辑，也避免把数据管线职责混入模型读出层。
"""

from __future__ import annotations

# argparse 保留一个可以从命令行直接运行的 main 自检入口。
import argparse
# Dict、Tuple 和 Union 用于描述训练辅助输出与两类返回形式。
from typing import Dict, Tuple, Union

# torch 提供张量运算、自动微分和测试输入。
import torch
# Tensor 是张量别名，nn 提供解码网络和预测头。
from torch import Tensor, nn
from torch.nn import functional as F


class CausalRankingDecoder(nn.Module):
    """把父节点感知表示解码为因果存在性、强度和排名分数。

    参数：
        embedding_dim: Parent Interaction 输出的嵌入宽度 E。
        hidden_dim: 共享证据解码网络的隐藏宽度。
        dropout: 共享解码网络使用的随机失活率。

    对每个候选因子 j，模块执行：

        r_j = f_dec(e_j)
        g_j = sigmoid(f_exist(r_j))
        a_j = softplus(f_effect(r_j))
        s_j = g_j * a_j

    这里将 tau 解释为非负的“因果效应强度”。如果未来任务需要预测带方向
    的干预效应，应另外增加 signed effect head，而不要直接用带符号效应
    替换 a_j，否则强负效应会在降序排名中被错误压到零效应之后。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        """初始化共享证据解码器、存在性头和效应强度头。"""

        # 调用 nn.Module 初始化，使 PyTorch 登记全部参数与子模块。
        super().__init__()
        # 嵌入宽度和隐藏宽度必须能够形成有效线性层。
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须为正整数。")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须为正整数。")
        # PyTorch Dropout 要求概率位于左闭右开的单位区间。
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须满足 0 <= dropout < 1。")

        # 保存维度配置，供输入检查和完整模型配置读取。
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 共享网络对每个因子独立使用同一套参数，因此不会依赖因子列编号。
        self.evidence_decoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        # 残差与归一化保留上游 parent-aware evidence 并稳定数值尺度。
        self.evidence_output_norm = nn.LayerNorm(embedding_dim)

        # 存在性头输出 logits；训练时应直接交给 BCEWithLogitsLoss。
        self.existence_head = nn.Linear(embedding_dim, 1)
        # 强度头先输出任意实数，再由 Softplus 映射为非负因果强度。
        self.effect_head = nn.Linear(embedding_dim, 1)
        self.effect_activation = nn.Softplus()

    def _validate_input(self, parent_features: Tensor) -> Tuple[Tensor, bool]:
        """检查输入并为单面板 [D,E] 临时增加 batch 维。"""

        # 二维输入表示一个面板；三维输入表示一批独立面板。
        remove_batch_dimension = parent_features.ndim == 2
        if parent_features.ndim not in (2, 3):
            raise ValueError("parent_features 必须是 [D,E] 或 [B,D,E]。")
        if remove_batch_dimension:
            parent_features = parent_features.unsqueeze(0)

        # 标准化后的输入必须具有非空 batch、候选因子和 embedding 维。
        batch_size, num_factors, embedding_dim = parent_features.shape
        if batch_size <= 0 or num_factors <= 0:
            raise ValueError("批量大小和候选因子数量都必须大于零。")
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"parent_features 最后一维应为 {self.embedding_dim}，"
                f"实际为 {embedding_dim}。"
            )
        # 神经网络读出层只能接收浮点表示。
        if not torch.is_floating_point(parent_features):
            raise TypeError("parent_features 必须是浮点张量。")
        # Decoder 不负责缺失值处理，应由上游数据管线提前完成。
        if not torch.isfinite(parent_features).all():
            raise ValueError("parent_features 中不能包含 NaN 或 Inf。")

        return parent_features, remove_batch_dimension

    def decode_evidence(self, parent_features: Tensor) -> Tensor:
        """共享解码每个 parent-aware representation，保持 [B,D,E]。"""

        # MLP 学习将上游通用表示转换成适合两个预测任务的共享证据空间。
        evidence_update = self.evidence_decoder(parent_features)
        # 残差路径避免最后一个模块无必要地覆盖已经形成的父节点感知信息。
        return self.evidence_output_norm(parent_features + evidence_update)

    def predict_existence(
        self,
        decoded_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """返回 [B,D] 的因果存在 logits 和对应概率。"""

        # 去掉线性层产生的末尾单元素维度，由 [B,D,1] 得到 [B,D]。
        existence_logits = self.existence_head(decoded_features).squeeze(-1)
        # 概率只用于 soft gate、评估与推理；BCE 训练应直接使用 logits。
        existence_probability = torch.sigmoid(existence_logits)
        return existence_logits, existence_probability

    def predict_effect(self, decoded_features: Tensor) -> Tensor:
        """预测 [B,D] 的非负因果效应强度。"""

        # 原始输出不受约束，Softplus 提供平滑且处处可导的非负映射。
        raw_effect = self.effect_head(decoded_features).squeeze(-1)
        return self.effect_activation(raw_effect)

    def compose_score(
        self,
        existence_probability: Tensor,
        effect_strength: Tensor,
    ) -> Tensor:
        """通过 soft causal gate 合成 [B,D] 最终排名分数。"""

        # 两个分支必须逐候选因子一一对应，禁止依赖广播隐藏形状错误。
        if existence_probability.shape != effect_strength.shape:
            raise ValueError("存在概率和效应强度必须具有完全相同的形状。")
        # g_j 抑制非父节点，a_j 区分被支持父节点之间的作用强弱。
        return existence_probability * effect_strength

    def compose_log_score(
        self,
        existence_logits: Tensor,
        effect_strength: Tensor,
    ) -> Tensor:
        """返回与最终乘积分数同序、但低概率区梯度更稳定的 log-score。"""

        if existence_logits.shape != effect_strength.shape:
            raise ValueError("存在 logits 和效应强度必须具有完全相同的形状。")
        # float32 避免 AMP 下很小的正效应在 log 前下溢。log 是单调变换，
        # 因而该分数的排序与 sigmoid(logit) * effect_strength 完全一致。
        return F.logsigmoid(existence_logits.float()) + torch.log(
            effect_strength.float().clamp_min(1e-12)
        )

    def forward(
        self,
        parent_features: Tensor,
        return_auxiliary: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """解码父节点证据，默认返回分数，训练时可返回全部中间预测。

        参数：
            parent_features: [B,D,E] 或单面板 [D,E]。
            return_auxiliary: False 时仅返回 scores；True 时返回训练字典。

        返回：
            默认是 [B,D] 或 [D] 的 scores。辅助字典中的全部逐因子
            张量也会按照输入形式保留或移除 batch 维。
        """

        # 在进入网络前统一形状并拒绝非法数值。
        parent_features, remove_batch_dimension = self._validate_input(
            parent_features
        )
        # 形成供两个任务头共享的证据表示。
        decoded_features = self.decode_evidence(parent_features)
        # 存在性分支同时保留数值稳定训练需要的 logits。
        existence_logits, existence_probability = self.predict_existence(
            decoded_features
        )
        # 强度分支输出非负连续量。
        effect_strength = self.predict_effect(decoded_features)
        # 最终因果排序分数由存在概率与效应强度共同决定。
        scores = self.compose_score(
            existence_probability,
            effect_strength,
        )
        log_scores = self.compose_log_score(existence_logits, effect_strength)

        # 单面板调用应移除内部临时增加的 batch 维。
        if remove_batch_dimension:
            scores = scores.squeeze(0)
            existence_logits = existence_logits.squeeze(0)
            existence_probability = existence_probability.squeeze(0)
            effect_strength = effect_strength.squeeze(0)
            log_scores = log_scores.squeeze(0)

        # 推理和普通评分只需要最终分数，避免外部依赖内部训练表示。
        if not return_auxiliary:
            return scores
        # 训练阶段返回各分支结果，由 trainer 在模块外组合监督损失。
        return {
            "scores": scores,
            "log_scores": log_scores,
            "existence_logits": existence_logits,
            "existence_probability": existence_probability,
            "effect_strength": effect_strength,
        }


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造 Decoder 自检入口的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="测试 CausalRank 因果排序解码器"
    )
    parser.add_argument("--batch-size", type=int, default=3, help="批量数量 B")
    parser.add_argument("--num-factors", type=int, default=7, help="候选因子数量 D")
    parser.add_argument("--embedding-dim", type=int, default=32, help="嵌入维度 E")
    parser.add_argument("--hidden-dim", type=int, default=64, help="隐藏维度")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout 概率")
    return parser


def main() -> None:
    """运行形状、辅助输出、置换等变性、变长输入和梯度测试。"""

    args = _build_argument_parser().parse_args()
    # 固定随机种子，使自检结果能够稳定复现。
    torch.manual_seed(42)
    model = CausalRankingDecoder(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    # 关闭 Dropout，保证单面板与置换比较只包含浮点舍入误差。
    model.eval()

    # 模拟 Parent Interaction 输出的 [B,D,E] 父节点感知表示。
    parent_features = torch.randn(
        args.batch_size,
        args.num_factors,
        args.embedding_dim,
    )
    # 默认接口只返回可直接排序的 [B,D] 分数。
    scores = model(parent_features)
    expected_shape = (args.batch_size, args.num_factors)
    assert scores.shape == expected_shape
    assert torch.isfinite(scores).all()
    assert (scores >= 0.0).all()

    # 训练接口额外返回两个任务头的内部结果。
    auxiliary = model(parent_features, return_auxiliary=True)
    assert isinstance(auxiliary, dict)
    assert set(auxiliary) == {
        "scores",
        "log_scores",
        "existence_logits",
        "existence_probability",
        "effect_strength",
    }
    for value in auxiliary.values():
        assert value.shape == expected_shape
        assert torch.isfinite(value).all()
    assert torch.allclose(scores, auxiliary["scores"])
    assert (auxiliary["existence_probability"] >= 0.0).all()
    assert (auxiliary["existence_probability"] <= 1.0).all()
    assert (auxiliary["effect_strength"] >= 0.0).all()
    assert torch.allclose(
        auxiliary["scores"],
        auxiliary["existence_probability"] * auxiliary["effect_strength"],
    )
    assert torch.allclose(
        auxiliary["log_scores"],
        torch.log(auxiliary["scores"].float()),
        atol=1e-6,
        rtol=1e-6,
    )

    # 单面板接口返回 [D]，并应与批量调用中的同一面板一致。
    single_scores = model(parent_features[0])
    assert single_scores.shape == (args.num_factors,)
    assert torch.allclose(single_scores, scores[0], atol=1e-6, rtol=1e-6)

    # 因子换序只应使输出发生相同换序，不能改变对应因子的分数。
    factor_permutation = torch.randperm(args.num_factors)
    permuted_scores = model(parent_features[:, factor_permutation])
    permutation_difference = (
        scores[:, factor_permutation] - permuted_scores
    ).abs().max().item()
    assert permutation_difference < 1e-6

    # 同一 Decoder 可以直接处理不同大小的紧凑候选集合，不需要 padding mask。
    variable_features = torch.randn(
        args.batch_size,
        args.num_factors + 2,
        args.embedding_dim,
    )
    variable_scores = model(variable_features)
    assert variable_scores.shape == (
        args.batch_size,
        args.num_factors + 2,
    )

    # 用三类模拟监督共同检查存在性头、强度头和共享解码器的计算图。
    model.train()
    training_features = parent_features.detach().clone().requires_grad_(True)
    training_outputs = model(training_features, return_auxiliary=True)
    causal_labels = torch.randint(
        low=0,
        high=2,
        size=expected_shape,
        dtype=training_features.dtype,
    )
    effect_targets = torch.rand(expected_shape)
    causal_loss = nn.functional.binary_cross_entropy_with_logits(
        training_outputs["existence_logits"],
        causal_labels,
    )
    # 示例中只在真实父节点位置监督效应强度。
    parent_weights = causal_labels
    effect_error = (
        training_outputs["effect_strength"] - effect_targets
    ).square()
    effect_loss = (
        effect_error * parent_weights
    ).sum() / parent_weights.sum().clamp_min(1.0)
    # 非常小的 score 项模拟最终排序损失对 soft gate 两条分支的联合梯度。
    mock_loss = (
        causal_loss
        + effect_loss
        + 0.01 * training_outputs["scores"].mean()
    )
    mock_loss.backward()

    # 所有参数都应参与常规前向计算，并产生有限梯度。
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    finite_gradients = all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    gradient_sum = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert not missing_gradients, f"存在未参与前向传播的参数：{missing_gradients}"
    assert finite_gradients, "模型参数梯度中出现 NaN 或 Inf。"
    assert gradient_sum > 0.0
    assert training_features.grad is not None
    assert torch.isfinite(training_features.grad).all()

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print("CausalRankingDecoder 测试通过")
    print(f"父节点感知输入形状：{tuple(parent_features.shape)}")
    print(f"最终排序分数形状：{tuple(scores.shape)}")
    print(f"因子置换最大误差：{permutation_difference:.3e}")
    print(f"全部参数梯度绝对值之和：{gradient_sum:.3e}")
    print(f"参数数量：{parameter_count:,}")


# 直接运行本文件时执行自检；作为 module 包导入时不会产生副作用。
if __name__ == "__main__":
    main()
