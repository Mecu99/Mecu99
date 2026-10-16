from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pricehunter.config import load_config


def test_relative_paths_resolve_next_to_the_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    path = config_dir / "pricehunter.yaml"
    absolute = tmp_path / "elsewhere" / "alerts.jsonl"
    path.write_text(
        "settings:\n  database: data/prices.db\n"
        "notifiers:\n"
        "  - {type: file, path: logs/alerts.jsonl}\n"
        f"  - {{type: file, path: '{absolute}'}}\n"
        "  - {type: console}\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.settings.database == str(config_dir / "data" / "prices.db")
    assert [n.path for n in config.notifiers] == [
        str(config_dir / "logs" / "alerts.jsonl"),
        str(absolute),
        None,
    ]


def test_environment_variables_are_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PH_HOOK", "https://hooks.example.com/abc")
    monkeypatch.delenv("PH_MISSING", raising=False)
    path = tmp_path / "c.yaml"
    path.write_text(
        "notifiers:\n"
        "  - {type: slack, url: '${PH_HOOK}'}\n"
        "  - {type: discord, url: '${PH_MISSING:-https://fallback.example/x}'}\n",
        encoding="utf-8",
    )
    assert [n.url for n in load_config(path).notifiers] == [
        "https://hooks.example.com/abc",
        "https://fallback.example/x",
    ]


def test_json_configs_and_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        '{"products": [{"sku": "A", "name": "Widget", "our_price": "9.99"}]}', encoding="utf-8"
    )
    assert load_config(path).products[0].sku == "A"
    path.write_text(
        '{"competitors": [{"name": "X", "base_url": "https://x"}, {"name": "X", "base_url": "https://y"}]}'
    )
    with pytest.raises(ValidationError, match="competitor names must be unique"):
        load_config(path)
