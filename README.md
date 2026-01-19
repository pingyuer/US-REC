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
命令会从配置加载参数，并在 `models_all/.../config.txt` 中写入最终设置，保持历史恢复脚本的习惯。

## 📖 文档导航

所有详细文档都在 `docs/` 目录中：
- 文档索引：`docs/README.md`

## 📁 项目结构

```
├── configs/              配置文件（YAML）
├── data/                数据加载和预处理
├── models/              模型定义和损失函数
├── trainers/            训练框架和 Hooks
├── utils/               工具函数和可视化
├── tests/               单元和集成测试
├── docs/                详细文档
├── main.py              训练入口
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
