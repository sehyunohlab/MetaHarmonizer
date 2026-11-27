"""OntoMapEngine: a caller-named ontology_source must agree with corpus_df codes."""
import pandas as pd
import pytest

from metaharmonizer.engine.ontology_mapping_engine import OntoMapEngine

UBERON = pd.DataFrame({"label": ["feces", "mouth"],
                       "obo_id": ["UBERON:0001988", "UBERON:0000165"]})


@pytest.fixture
def no_tables(monkeypatch):
    # concept tables would call OLS; source resolution happens before them
    monkeypatch.setattr(OntoMapEngine, "_ensure_concept_tables", lambda self, df: None)


def _engine(**kw):
    return OntoMapEngine("bodysite", ["stool"], corpus_df=UBERON, corpus_hash="t", **kw)


def test_contradicting_source_raises(no_tables):
    with pytest.raises(ValueError, match="contradicts corpus_df, whose codes are uberon"):
        _engine(ontology_source="mondo")


def test_omitted_source_is_read_from_codes(no_tables):
    assert _engine()._ontology_source == "uberon"


def test_matching_source_is_accepted(no_tables):
    assert _engine(ontology_source="UBERON")._ontology_source == "uberon"


def test_local_corpus_keeps_named_source(no_tables):
    local = pd.DataFrame({"label": ["Female", "Male"], "obo_id": ["LOCAL:sex_0", "LOCAL:sex_1"]})
    e = OntoMapEngine("sex", ["f"], corpus_df=local, corpus_hash="t", ontology_source="local")
    assert e._ontology_source == "local" and e._local_corpus


def test_same_source_accepts_prefix_names():
    assert OntoMapEngine._same_source("orphanet", "ordo")
    assert OntoMapEngine._same_source("MONDO", "mondo")
    assert not OntoMapEngine._same_source("mondo", "ncit")
