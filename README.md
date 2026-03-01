# Medical Image Segmentation Framework

一个配置驱动的医学图像分割训练框架，基于 PyTorch 和 OmegaConf。

## 🚀 快速开始

### 1. 环境配置
```bash
# 创建虚拟环境
conda create -n seg python=3.10
conda activate seg

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行项目
```bash
# 运行测试（离线默认）
pytest -q

# 开始训练
python main.py --config configs/demo_tui.yml

```

## 🎯 Free-hand 分割训练

入口统一到 `main_rec.py`，配置文件位于 `configs/`。运行：
```bash
python main_rec.py --config configs/demo_rec24_ete.yml
python main_rec.py --config configs/demo_rec24_meta.yml
```

### 快速评测 & 诊断
```bash
# 抽样评测（1 scan, 10 frames）
python main_rec.py --config configs/demo_rec24_ete.yml --eval-only --max-scans 1 --max-frames 10

# Dry-run：验证组件构建 + 数据可读
python main_rec.py --config configs/demo_rec24_ete.yml --dry-run

# 统一评测入口
python -m eval.run_eval --config configs/demo_rec24_ete.yml --eval-only --export-json
```

### Smoke Tests（抽样测试，< 5 分钟）
```bash
# 运行所有 smoke tests
pytest -m smoke

# 运行全量 / 慢速测试（可选）
pytest -m slow

# 只跑 compose 方向校验
pytest tests/smoke/test_smoke_compose_global_direction.py -v
```

### 关键架构说明

| 关注点 | 唯一入口 |
|--------|----------|
| Metrics（所有指标） | `trainers/metrics/` — 其他模块仅 import 此处 |
| Global/Local compose | `metrics/compose.py` — 权威实现 |
| 评测 & 导出 | `eval/run_eval.py` + `eval/export.py` |
| 可视化 | `viz/pose_curve.py`, `viz/drift_curve.py`, `viz/recon_slices.py` |
| 采样限制 | config `data.max_scans` / `data.max_frames_per_scan` / `eval.max_scans` |

## 📖 文档导航

所有详细文档都在 `docs/` 目录中：
- 文档索引：`docs/README.md`

## 📁 项目结构

```
├── configs/              配置文件（YAML）
├── data/                数据加载和预处理
├── eval/                统一评测与导出入口
│   ├── run_eval.py      单入口 --eval-only / --dry-run
│   ├── builder.py       评测组件构建
│   └── export.py        per-scan JSON / NPZ 导出
├── metrics/             全局指标（权威 compose 实现）
│   ├── compose.py       compose_global_from_local / local_from_global
│   └── __init__.py      统一 re-export
├── models/              模型定义和损失函数
├── trainers/            训练框架、Hooks、Metrics
│   └── metrics/         指标具体实现（pose/trajectory/volume/tusrec…）
├── utils/               工具函数和可视化
├── viz/                 可视化与诊断导出
│   ├── pose_curve.py    平移/旋转 vs 帧号
│   ├── drift_curve.py   GPE vs 帧号 drift 曲线
│   └── recon_slices.py  重建体切片导出
├── tests/               单元和集成测试
│   └── smoke/           快速抽样 smoke tests（pytest -m smoke）
├── main.py              分割训练入口
├── main_rec.py          TUS-REC 训练/评测 CLI
└── requirements.txt     依赖包列表
```

更多详情见 `docs/core.md`

## 🎯 核心特性

✅ **配置驱动** - YAML 配置，无需修改代码  
✅ **工厂模式** - 快速构建数据集、模型、优化器  
✅ **灵活架构** - 支持多种分割网络 (UNet, UNext, HJUNet)  
✅ **完整测试** - 18 个单元和集成测试  
✅ **优秀文档** - 面向不同角色的多层次文档  

## 🔧 支持的模型

- **UNet** - 经典分割架构
- **UNext** - MLP 快速分割
- **HJUNet** - 概率分割网络

## 📊 支持的数据集

- BUSI (乳腺超声)
- ISIC2017/2018 (皮肤病变)
- Synapse (多器官)
- 自定义数据集

## 📝 项目信息

更新时间：2025-11-24  
Python 版本：3.10+  
PyTorch 版本：1.7+  
测试状态：✅ pytest 通过  

---

**需要帮助？** 查看 `docs/README.md`
