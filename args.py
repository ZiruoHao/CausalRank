"""CausalRank 训练命令行参数。"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析并校验训练、数据和模型配置。"""

    parser = argparse.ArgumentParser(description="CausalRank")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="测试集目录；默认读取 --train-data 目录中的 test split",
    )
    parser.add_argument("--replay-data", type=Path, default=None)
    parser.add_argument("--replay-ratio", type=float, default=0.30)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/causalrank"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    # 完整面板在双 24 GiB GPU 上实测可稳定完成整轮的保守 micro-batch。
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--effect-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=25.25,
        help="固定 BCE 正类权重；当前128-episode训练集的全局负正比约为25.25",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="学习率从 base_lr/warmup 线性升至 base_lr 的 epoch 数；0 表示关闭",
    )
    parser.add_argument(
        "--min-learning-rate-ratio",
        type=float,
        default=0.1,
        help="cosine 末端学习率相对 --learning-rate 的比例",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cached-shards", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument(
        "--no-validate-shards",
        dest="validate_shards",
        action="store_false",
    )
    parser.set_defaults(validate_shards=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="主设备或横截面并行设备，例如 0、cuda:0、0,1、cpu、auto",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--cross-section-chunk-size",
        type=int,
        default=120,
        help="每次编码的横截面表数量；0 表示一次处理全部 B*T",
    )
    parser.add_argument(
        "--cross-section-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="对横截面编码启用激活重计算，以计算时间换取更低峰值显存",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    # 同时控制 epoch 级日志和 checkpoint 周期，默认每 10 个 epoch 执行一次。
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--num-inducing-points", type=int, default=32)
    parser.add_argument("--num-seed-vectors", type=int, default=4)
    parser.add_argument("--cross-row-blocks", type=int, default=4)
    parser.add_argument("--cross-column-blocks", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--max-time-steps", type=int, default=256)
    parser.add_argument("--parent-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--conditional-ridge",
        type=float,
        default=1e-2,
        help="显式条件增量证据中岭回归与偏相关协方差的正则强度",
    )
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    if not 0.0 <= args.replay_ratio < 1.0:
        parser.error("--replay-ratio 必须满足 0 <= ratio < 1")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs 和 --batch-size 必须为正数")
    if args.log_every <= 0:
        parser.error("--log-every 必须为正数")
    if args.cross_section_chunk_size < 0:
        parser.error("--cross-section-chunk-size 不能为负数")
    if args.positive_class_weight < 0.0:
        parser.error("--positive-class-weight 不能为负数")
    if args.conditional_ridge <= 0.0:
        parser.error("--conditional-ridge 必须为正数")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("--warmup-epochs 必须满足 0 <= warmup < epochs")
    if not 0.0 <= args.min_learning_rate_ratio <= 1.0:
        parser.error("--min-learning-rate-ratio 必须满足 0 <= ratio <= 1")
    return args
