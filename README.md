# CausalRank

CausalRank 的模型、数据生成和训练代码。Git 仓库只保存代码、文档与环境定义；
原始数据、生成的 episode、训练结果和 checkpoint 均保留在本地或通过独立存储迁移。

## 环境

当前验证环境为 Python 3.11、PyTorch 2.5.1 和 CUDA 12.1。使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate causalrank
python tests/test_ranking_metrics.py
```

`pytorch-cuda=12.1` 需要新服务器安装兼容的 NVIDIA 驱动。若只使用 CPU，请从
`environment.yml` 删除 `pytorch-cuda`，并根据目标平台安装 PyTorch CPU 构建。
环境文件固定了与 PyTorch 2.5.1 兼容的 MKL/Intel OpenMP 版本；不要将 MKL
单独升级到 2024.1 或更高版本，否则导入 PyTorch 时可能缺少
`iJIT_NotifyEvent` 符号。

## 不进入 Git 的目录

- `data/`：GKX 原始数据及缓存；仅跟踪 `data/__init__.py`。
- `data_generation/*/`：各生成器产生的数据集；生成器根目录下的 Python 脚本会跟踪。
- `runs/`：训练日志、指标和 checkpoint。

克隆仓库后应根据需要重新生成数据，或使用 `rsync` 等工具单独迁移上述目录。

## 数据准备

半合成和全合成生成器默认在以下位置查找 GKX 数据：

```text
data/GKX Characteristics Data/datashare.csv
```

也可以通过 `--input-csv` 指定其他位置。查看完整参数：

```bash
python data_generation/generate_semi_synthetic.py --help
python data_generation/generate_full_synthetic.py --help
```

示例：

```bash
python data_generation/generate_semi_synthetic.py \
  --input-csv "/path/to/datashare.csv" \
  --output-dir data_generation/semi_synthetic

python data_generation/generate_full_synthetic.py \
  --input-csv "/path/to/datashare.csv" \
  --output-dir data_generation/full_synthetic
```

## 训练

训练目录必须包含生成器写出的 `manifest.json` 以及 `train/`、`validation/`、
`test/` 分片。基本命令：

```bash
python train.py \
  --train-data data_generation/full_synthetic \
  --output-dir runs/causalrank \
  --device auto \
  --amp
```

查看全部训练参数：

```bash
python train.py --help
```

## 迁移数据（可选）

Git 不用于迁移数据与结果。如果新服务器能够 SSH 访问旧服务器，可以在新服务器上
按需执行：

```bash
rsync -avhP OLD_USER@OLD_HOST:/home/dell/CausalRank/data/ ./data/
rsync -avhP OLD_USER@OLD_HOST:/home/dell/CausalRank/data_generation/ ./data_generation/
rsync -avhP OLD_USER@OLD_HOST:/home/dell/CausalRank/runs/ ./runs/
```

只迁移代码时无需执行这些命令。
