"""Tests for YAML config loading and defaults."""

from craft_tabqa.config import load_config


def test_load_config_applies_yaml_and_keeps_defaults(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
data:
  dataset: custom
  corpus_file: /tmp/corpus.jsonl
stage2:
  mode: mini_table
  top_k: 50
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.data.dataset == "custom"
    assert cfg.data.corpus_file == "/tmp/corpus.jsonl"
    assert cfg.stage2.mode == "mini_table"
    assert cfg.stage2.top_k == 50
    # untouched defaults
    assert cfg.stage1.top_k == 5000
    assert cfg.stage2.top_k_rows == 5


def test_default_stage2_mode_is_representative_row(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("data:\n  dataset: custom\n", encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg.stage2.mode == "representative_row"
