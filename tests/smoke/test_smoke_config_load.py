"""Smoke test 1: config loading and field validation.

Verifies that ``configs/demo_rec24_ete.yml`` loads without error and
contains all mandatory top-level sections and key fields.
"""

import pytest
from pathlib import Path
from omegaconf import OmegaConf


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "demo_rec24_ete.yml"


@pytest.mark.smoke
class TestSmokeConfigLoad:
    """Validate demo_rec24_ete.yml structure."""

    @pytest.fixture(autouse=True)
    def _load(self):
        assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
        self.cfg = OmegaConf.load(str(CONFIG_PATH))

    def test_top_level_sections(self):
        for section in ("model", "dataset", "trainer", "loss", "optimizer"):
            assert section in self.cfg, f"Missing top-level section: {section}"

    def test_model_fields(self):
        m = self.cfg.model
        assert m.get("name"), "model.name missing"
        assert m.get("pred_type"), "model.pred_type missing"
        assert m.get("label_type"), "model.label_type missing"

    def test_dataset_fields(self):
        d = self.cfg.dataset
        assert d.get("name"), "dataset.name missing"

    def test_trainer_fields(self):
        t = self.cfg.trainer
        assert t.get("max_epochs") is not None, "trainer.max_epochs missing"

    def test_paths_section(self):
        assert "paths" in self.cfg, "Missing paths section"
        p = self.cfg.paths
        assert p.get("output_dir"), "paths.output_dir missing"

    def test_sampling_fields(self):
        """Sampling limit fields must exist (even if null)."""
        assert "data" in self.cfg, "Missing data section"
        d = self.cfg.data
        assert "max_scans" in d
        assert "max_frames_per_scan" in d

    def test_eval_section(self):
        assert "eval" in self.cfg, "Missing eval section"
        assert "max_scans" in self.cfg.eval

    def test_seed_field(self):
        assert "seed" in self.cfg, "Missing seed field"
