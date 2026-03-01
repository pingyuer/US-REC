import pytest
pytestmark = pytest.mark.integration

#!/usr/bin/env python
"""
Quick test script to verify main.py can run with a demo config.
This creates a minimal dummy dataset for testing.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

# Create a simple dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, num_samples=4, img_size=32, num_classes=2, **kwargs):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return random image and mask
        image = torch.randn(3, self.img_size, self.img_size)
        mask = torch.randint(0, self.num_classes, (self.img_size, self.img_size))
        return {"image": image, "mask": mask}


def test_main():
    """Test main.py with a demo config (pytest compatible)."""
    print("\n" + "="*70)
    print("Testing main.py with demo config")
    print("="*70 + "\n")
    
    # Test 1: Load config
    print("[Test 1] Loading config...")
    config_path = project_root / "configs" / "demo.yml"
    cfg = OmegaConf.load(str(config_path))
    assert cfg is not None, "Config loading failed"
    print(f"✅ Config loaded\n")
    
    # Test 2: Import and build components
    print("[Test 2] Importing modules...")
    try:
        from models import build_model
        from trainers.builder import build_optimizer
        from trainers.trainer import Trainer
        print(f"✅ Imports successful\n")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")
    
    # Test 3: Build model
    print("[Test 3] Building model...")
    try:
        model = build_model(cfg.model)
        assert model is not None, "Model building returned None"
        print(f"✅ Model built: {model.__class__.__name__}\n")
    except Exception as e:
        pytest.fail(f"Model building failed: {e}")
    
    # Test 4: Build optimizer
    print("[Test 4] Building optimizer...")
    try:
        optimizer = build_optimizer(cfg.optimizer, model)
        assert optimizer is not None, "Optimizer building returned None"
        print(f"✅ Optimizer built\n")
    except Exception as e:
        pytest.fail(f"Optimizer building failed: {e}")
    
    # Test 5: Build trainer with dummy dataloader
    print("[Test 5] Building trainer with dummy dataset...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy dataloaders
        dummy_dataset = DummyDataset(num_samples=4, img_size=32, num_classes=2)
        train_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=False)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device
        )
        assert trainer is not None, "Trainer building returned None"
        print(f"✅ Trainer built\n")
    except Exception as e:
        pytest.fail(f"Trainer building failed: {e}")
    
    # Test 6: Run one training step
    print("[Test 6] Running 1 epoch of training...")
    try:
        # Just run the first epoch
        cfg.trainer.max_epochs = 1
        trainer.train()
        print(f"✅ Training completed\n")
    except Exception as e:
        pytest.fail(f"Training failed: {e}")
    
    print("="*70)
    print("✨ All tests passed! main.py is ready to run.")
    print("="*70 + "\n")


def run_integration_test():
    """Standalone function for running tests directly (non-pytest)."""
    print("\n" + "="*70)
    print("Testing main.py with demo config")
    print("="*70 + "\n")
    
    # Test 1: Load config
    print("[Test 1] Loading config...")
    config_path = project_root / "configs" / "demo.yml"
    cfg = OmegaConf.load(str(config_path))
    print(f"✅ Config loaded\n")
    
    # Test 2: Import and build components
    print("[Test 2] Importing modules...")
    try:
        from models import build_model
        from trainers.builder import build_optimizer
        from trainers.trainer import Trainer
        print(f"✅ Imports successful\n")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 3: Build model
    print("[Test 3] Building model...")
    try:
        model = build_model(cfg.model)
        print(f"✅ Model built: {model.__class__.__name__}\n")
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        return False
    
    # Test 4: Build optimizer
    print("[Test 4] Building optimizer...")
    try:
        optimizer = build_optimizer(cfg.optimizer, model)
        print(f"✅ Optimizer built\n")
    except Exception as e:
        print(f"❌ Optimizer building failed: {e}")
        return False
    
    # Test 5: Build trainer with dummy dataloader
    print("[Test 5] Building trainer with dummy dataset...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy dataloaders
        dummy_dataset = DummyDataset(num_samples=4, img_size=32, num_classes=2)
        train_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=False)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device
        )
        print(f"✅ Trainer built\n")
    except Exception as e:
        print(f"❌ Trainer building failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Run one training step
    print("[Test 6] Running 1 epoch of training...")
    try:
        # Just run the first epoch
        cfg.trainer.max_epochs = 1
        trainer.train()
        print(f"✅ Training completed\n")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*70)
    print("✨ All tests passed! main.py is ready to run.")
    print("="*70 + "\n")
    print("Usage:")
    print("  python main.py --config configs/base.yml")
    print("  python main.py --config configs/demo.yml trainer.max_epochs=5")
    print()
    
    return True


if __name__ == "__main__":
    # When run directly, use the standalone function
    success = run_integration_test()
    sys.exit(0 if success else 1)
