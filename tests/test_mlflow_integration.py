#!/usr/bin/env python
"""
MLflow 集成测试脚本

验证:
1. MLflow 连接
2. 实验创建
3. 运行记录
4. 指标日志
5. 工件保存
"""

import os
import pytest

pytestmark = pytest.mark.integration

# Default skip: requires a reachable MLflow server.
if os.getenv("RUN_MLFLOW_INTEGRATION", "0") != "1":
    pytest.skip("Set RUN_MLFLOW_INTEGRATION=1 to run MLflow server integration tests", allow_module_level=True)

import torch
import torch.nn as nn
import tempfile
import sys
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mlflow_manager import MLflowManager


def create_dummy_config():
    """创建测试配置"""
    cfg = OmegaConf.create({
        "model": {
            "name": "test_unet",
            "num_classes": 2,
        },
        "dataset": {
            "name": "data.datasets.seg_dataset.DummyDataset",
            "img_dir": {"train": "/tmp/train", "val": "/tmp/val"},
        },
        "optimizer": {
            "type": "adamw",
            "lr": 0.0001,
            "weight_decay": 0.01,
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "mlflow_integration_test",
            "run_name": None,  # 自动生成
            "tags": {
                "test": "true",
                "phase": "testing",
            },
            "logging": {
                "log_params": True,
                "log_train_loss": True,
                "log_val_metrics": True,
                "log_learning_rate": True,
                "log_images": True,
                "log_interval": 10,
            },
            "artifacts": {
                "save_best_model": True,
                "best_model_metric": "val_loss",
                "best_model_mode": "min",
                "save_config": True,
            },
        },
    })
    return cfg


def test_mlflow_initialization():
    """测试 MLflow 初始化"""
    print("\n" + "="*70)
    print("[Test 1] MLflow 初始化")
    print("="*70)
    
    cfg = create_dummy_config()
    
    try:
        mlflow = MLflowManager(cfg)
        
        if mlflow.enabled:
            print("✅ MLflow 连接成功")
            print(f"   - Tracking URI: {MLflowManager.get_tracking_uri()}")
            print(f"   - Experiment: {cfg.mlflow.experiment_name}")
            print(f"   - Run ID: {mlflow.run_id[:8]}...")
            return mlflow
        else:
            print("⚠️  MLflow 已禁用（MLflow 服务器可能未运行）")
            return mlflow
    
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("   💡 提示: 确保 MLflow 服务器正在运行")
        print("   运行: mlflow ui --host 0.0.0.0 --port 5000")
        return None


def test_metric_logging(mlflow):
    """测试指标记录"""
    print("\n" + "="*70)
    print("[Test 2] 指标记录")
    print("="*70)
    
    if not mlflow or not mlflow.enabled:
        print("⏭️  跳过（MLflow 已禁用）")
        return
    
    try:
        # 记录训练指标
        for step in range(1, 51, 10):
            loss = 1.0 - step * 0.01  # 模拟递减的损失
            lr = 0.0001
            mlflow.log_train_step(loss=loss, step=step, learning_rate=lr)
            print(f"   ✅ Step {step}: loss={loss:.4f}, lr={lr:.4e}")
        
        # 记录验证指标
        for epoch in range(1, 6):
            metrics = {
                "loss": 0.5 - epoch * 0.05,
                "mIoU": 0.5 + epoch * 0.05,
                "dice": 0.5 + epoch * 0.06,
            }
            mlflow.log_val_metrics(metrics, epoch=epoch)
            print(f"   ✅ Epoch {epoch}: metrics={list(metrics.keys())}")
        
        print("✅ 指标记录成功")
    
    except Exception as e:
        print(f"❌ 记录失败: {e}")


def test_model_saving(mlflow):
    """测试模型保存"""
    print("\n" + "="*70)
    print("[Test 3] 模型保存")
    print("="*70)
    
    if not mlflow or not mlflow.enabled:
        print("⏭️  跳过（MLflow 已禁用）")
        return
    
    try:
        # 创建虚拟模型
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        
        # 保存最佳模型
        mlflow.save_best_model(
            model=model,
            metric_value=0.75,
            epoch=5,
            metric_name="mIoU"
        )
        print("✅ 最佳模型保存成功")
        
        # 保存检查点
        optimizer = torch.optim.Adam(model.parameters())
        mlflow.save_last_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5
        )
        print("✅ 检查点保存成功")
    
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def test_config_saving(mlflow):
    """测试配置保存"""
    print("\n" + "="*70)
    print("[Test 4] 配置保存")
    print("="*70)
    
    if not mlflow or not mlflow.enabled:
        print("⏭️  跳过（MLflow 已禁用）")
        return
    
    try:
        mlflow.save_config()
        print("✅ 配置文件保存成功")
    
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def test_image_logging(mlflow):
    """测试图像记录"""
    print("\n" + "="*70)
    print("[Test 5] 验证图像记录")
    print("="*70)
    
    if not mlflow or not mlflow.enabled:
        print("⏭️  跳过（MLflow 已禁用）")
        return
    
    try:
        import numpy as np
        
        # 创建虚拟数据
        batch_size = 3
        h, w = 128, 128
        
        images = [torch.rand(3, h, w) for _ in range(batch_size)]
        masks = [torch.randint(0, 2, (h, w)) for _ in range(batch_size)]
        predictions = [torch.randint(0, 2, (h, w)) for _ in range(batch_size)]
        
        mlflow.log_validation_images(
            images=images,
            masks=masks,
            predictions=predictions,
            epoch=1,
            max_images=3
        )
        print(f"✅ {batch_size} 张验证图像记录成功")
    
    except Exception as e:
        print(f"❌ 记录失败: {e}")


def test_run_completion(mlflow):
    """测试运行完成"""
    print("\n" + "="*70)
    print("[Test 6] 运行完成")
    print("="*70)
    
    if not mlflow or not mlflow.enabled:
        print("⏭️  跳过（MLflow 已禁用）")
        return
    
    try:
        mlflow.end_run(status="FINISHED")
        print("✅ 运行已完成")
    
    except Exception as e:
        print(f"❌ 完成失败: {e}")


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🧪 MLflow 集成测试")
    print("="*70)
    
    # Test 1: 初始化
    mlflow = test_mlflow_initialization()
    
    if mlflow is None or not mlflow.enabled:
        print("\n" + "="*70)
        print("⚠️  MLflow 服务器未运行")
        print("="*70)
        print("\n请启动 MLflow 服务器:")
        print("  mlflow ui --host 0.0.0.0 --port 5000")
        print("\n然后重新运行此测试脚本。\n")
        return
    
    # Test 2-6: 其他测试
    test_metric_logging(mlflow)
    test_model_saving(mlflow)
    test_config_saving(mlflow)
    test_image_logging(mlflow)
    test_run_completion(mlflow)
    
    # 总结
    print("\n" + "="*70)
    print("✨ 所有测试完成！")
    print("="*70)
    print("\n📊 访问 MLflow UI:")
    print(f"   {MLflowManager.get_tracking_uri()}")
    print("\n可以查看:")
    print("   - 实验: mlflow_integration_test")
    print("   - 所有记录的指标、工件等")
    print("\n")


if __name__ == "__main__":
    main()
