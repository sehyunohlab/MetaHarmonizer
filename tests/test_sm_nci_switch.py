"""Tests for the SchemaMapper NCI EVS ontology switch (``nci_ontology``)."""
from unittest.mock import MagicMock

import pytest

from metaharmonizer.models.schema_mapper import engine as engine_mod
from metaharmonizer.models.schema_mapper.engine import SchemaMapEngine
from metaharmonizer.settings import Settings, _to_bool


# ── Settings resolution ───────────────────────────────────────────────────────

def test_default_is_on():
    assert Settings().nci_ontology is True


def test_env_disables(monkeypatch):
    monkeypatch.setenv("SM_NCI_ONTOLOGY", "0")
    assert Settings.resolve().nci_ontology is False


def test_override_beats_env(monkeypatch):
    monkeypatch.setenv("SM_NCI_ONTOLOGY", "off")
    assert Settings.resolve(nci_ontology=True).nci_ontology is True


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("On", True), ("yes", True), (True, True),
    ("0", False), ("FALSE", False), ("off", False), ("no", False), (False, False),
])
def test_to_bool(val, expected):
    assert _to_bool(val) is expected


def test_to_bool_rejects_garbage():
    with pytest.raises(ValueError):
        _to_bool("maybe")


# ── Stage-2 cascade wiring ────────────────────────────────────────────────────

def _fake_engine(ontology):
    eng = MagicMock()
    eng.settings = Settings()
    eng.stage2_min_score = None
    eng.is_col_numeric.return_value = False
    eng.ontology = ontology
    eng._run_cascade.side_effect = lambda col, stage, strategies, **kw: [s.name for s in strategies]
    return eng


def test_stage2_without_ontology_runs_value_dict_only():
    names = SchemaMapEngine.stage2_match(_fake_engine(None), "sex")
    assert names == ["value"]


def test_stage2_with_ontology_runs_both():
    names = SchemaMapEngine.stage2_match(_fake_engine(MagicMock()), "sex")
    assert names == ["value", "ontology"]


# ── Engine construction: off means no NCI client and no ontology matcher ────

@pytest.fixture
def light_engine(monkeypatch, tmp_path):
    """Patch out model loading and the NCI client so construction is cheap."""
    nci = MagicMock(name="NCIClientSync")
    monkeypatch.setattr(engine_mod, "NCIClientSync", nci)
    monkeypatch.setattr(engine_mod, "load_sentence_transformer_cached", lambda *a, **k: MagicMock())
    monkeypatch.setattr(SchemaMapEngine, "_load_dictionaries", lambda self: setattr(self, "has_alias_dict", False))
    monkeypatch.setattr(engine_mod, "OUTPUT_DIR", str(tmp_path))
    src = tmp_path / "in.csv"
    src.write_text("sex,age\nmale,40\n")
    return nci, str(src)


def test_engine_off_creates_no_client(light_engine):
    nci, src = light_engine
    eng = SchemaMapEngine(src, nci_ontology=False)
    assert eng.nci_ontology is False
    assert eng.nci_client is None and eng.ontology is None
    nci.assert_not_called()


def test_engine_default_on(light_engine):
    nci, src = light_engine
    eng = SchemaMapEngine(src)
    assert eng.nci_ontology is True and eng.ontology is not None
    nci.assert_called_once()


def test_engine_follows_settings(light_engine):
    nci, src = light_engine
    eng = SchemaMapEngine(src, settings=Settings.resolve(nci_ontology=False))
    assert eng.nci_client is None
    nci.assert_not_called()
