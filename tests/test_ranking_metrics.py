"""CausalRank 排序评估指标的确定性单元测试。

测试使用可以手工计算的小向量，不需要加载模型或数据集。这样可以单独验证
AUPRC、AUROC、Top-k 与候选掩码的定义，避免“训练正常但指标实现错误”。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import compute_episode_ranking_statistics


def assert_close(actual: torch.Tensor, expected: float, name: str) -> None:
    """用清晰的字段名报告数值断言失败。"""

    if not torch.isclose(actual.float(), torch.tensor(expected), atol=1e-6):
        raise AssertionError(f"{name}: actual={float(actual)}, expected={expected}")


def main() -> None:
    """验证完美排序、部分排序、宏平均以及无效候选屏蔽。"""

    # episode 0 是完美排序；episode 1 的正类位于第 1、3 名。
    # 对 episode 1：AP=(1 + 2/3)/2=5/6，AUROC=3/4，Top-2 P=R=1/2。
    scores = torch.tensor([
        [0.9, 0.2, 0.8, 0.1],
        [0.9, 0.8, 0.7, 0.6],
    ])
    labels = torch.tensor([
        [True, False, True, False],
        [True, False, True, False],
    ])
    valid = torch.ones_like(labels)
    metrics = compute_episode_ranking_statistics(scores, labels, valid, top_k=2)

    assert_close(metrics["ranking_auprc_sum"], 1.0 + 5.0 / 6.0, "AUPRC sum")
    assert_close(metrics["ranking_auroc_sum"], 1.0 + 3.0 / 4.0, "AUROC sum")
    assert_close(metrics["ranking_auc_episode_count"], 2.0, "AUC episode count")
    assert_close(metrics["top20_precision_sum"], 1.5, "Top-k precision sum")
    assert_close(metrics["top20_recall_sum"], 1.5, "Top-k recall sum")
    assert_close(metrics["top20_episode_count"], 2.0, "Top-k episode count")

    # 一个无效候选即使拥有最高分也必须被完全排除。剩余有效候选中正类排名
    # 第一，因此四项指标均应为 1。
    masked_scores = torch.tensor([[100.0, 0.9, 0.1]])
    masked_labels = torch.tensor([[False, True, False]])
    masked_valid = torch.tensor([[False, True, True]])
    masked_metrics = compute_episode_ranking_statistics(
        masked_scores, masked_labels, masked_valid, top_k=1
    )
    for name in (
        "ranking_auprc_sum",
        "ranking_auroc_sum",
        "top20_precision_sum",
        "top20_recall_sum",
    ):
        assert_close(masked_metrics[name], 1.0, name)

    print("PASS: ranking AUPRC/AUROC/Top-k metrics", flush=True)


if __name__ == "__main__":
    main()
