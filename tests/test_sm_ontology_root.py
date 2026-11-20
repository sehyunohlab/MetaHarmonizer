"""Tests for the target schema's ontology_root column and the stage-2 gate."""
import pandas as pd
import pytest

from metaharmonizer.models.schema_mapper.engine import SchemaMapEngine, _parse_ontology_roots
from metaharmonizer.models.schema_mapper.matchers.base import MatchStrategy
from metaharmonizer.utils.ncit_match_utils import NCIT_DICT, NCIClientSync


# ── _parse_ontology_roots ──────────────────────────────────────────────────────

class TestParseOntologyRoots:
    def test_no_column_returns_none(self):
        df = pd.DataFrame({"field_name": ["disease"], "is_numeric_field": ["no"]})
        assert _parse_ontology_roots(df) is None

    def test_ncit_curies_parsed(self):
        df = pd.DataFrame({
            "field_name": ["disease", "disease_details", "treatment", "sex"],
            "ontology_root": ["NCIT:C2991", "NCIT:C2991", "ncit:C1909", None],
        })
        assert _parse_ontology_roots(df) == {
            "C2991": ["disease", "disease_details"],
            "C1909": ["treatment"],
        }

    def test_multiple_roots_per_field(self):
        df = pd.DataFrame({"field_name": ["disease"],
                           "ontology_root": ["NCIT:C2991 | NCIT:C3262"]})
        assert _parse_ontology_roots(df) == {"C2991": ["disease"], "C3262": ["disease"]}

    def test_non_ncit_and_bare_roots_skipped(self):
        # UBERON is not supported yet (issue #104); a bare code has no prefix
        df = pd.DataFrame({"field_name": ["body_site", "disease"],
                           "ontology_root": ["UBERON:0001062", "NCIT:C2991|C3262"]})
        assert _parse_ontology_roots(df) == {"C2991": ["disease"]}

    def test_column_present_but_empty_gives_empty_table(self):
        df = pd.DataFrame({"field_name": ["sex"], "ontology_root": [None]})
        assert _parse_ontology_roots(df) == {}


# ── NCIClientSync root table ───────────────────────────────────────────────────

class TestNCIClientRootFields:
    def test_default_is_builtin_table(self):
        client = NCIClientSync()
        assert client.root_fields == {c: [f] for c, f in NCIT_DICT.items()}
        assert set(client.desc_sets) == set(NCIT_DICT)

    def test_custom_roots_classify_to_schema_fields(self):
        client = NCIClientSync(root_fields={"C1909": ["treatment"],
                                            "C2991": ["disease", "disease_details"]})
        assert set(client.desc_sets) == {"C1909", "C2991"}
        assert client.classify_code("C1909") == ["treatment"]
        assert client.classify_code("C2991") == ["disease", "disease_details"]

    def test_custom_roots_descendant_from_bundled_file(self):
        client = NCIClientSync(root_fields={"C1909": ["treatment"]})
        child = next(iter(client.desc_sets["C1909"] - {"C1909"}))
        assert client.classify_code(child) == ["treatment"]


# ── stage-2 gate in _run_cascade ───────────────────────────────────────────────

def _cascade_engine():
    eng = SchemaMapEngine.__new__(SchemaMapEngine)  # no models, no I/O
    eng.top_k = 3
    return eng


def _strategy(name, matches, threshold=0.2):
    return MatchStrategy(name, lambda col: matches, threshold=threshold)


class TestStage2Gate:
    def test_no_gate_keeps_weak_match(self):
        out = _cascade_engine()._run_cascade(
            "body_subsite", "stage2", [_strategy("ontology", [("body_site", 0.27, "x")])])
        assert out["match1"] == "body_site"

    def test_gate_declines_match_above_strategy_threshold(self):
        # 0.27 >= 0.2 used to return before the gate was checked
        out = _cascade_engine()._run_cascade(
            "body_subsite", "stage2", [_strategy("ontology", [("body_site", 0.27, "x")])],
            min_score=0.5)
        assert out == {}

    def test_gate_accepts_match_above_gate(self):
        out = _cascade_engine()._run_cascade(
            "col", "stage2", [_strategy("ontology", [("disease", 0.76, "x")])], min_score=0.5)
        assert out["match1"] == "disease"

    def test_gate_moves_on_to_next_strategy(self):
        out = _cascade_engine()._run_cascade(
            "col", "stage2",
            [_strategy("value", [("sex", 0.4, "x")]),
             _strategy("ontology", [("disease", 0.6, "x")])],
            min_score=0.5)
        assert out["method"] == "ontology" and out["match1"] == "disease"

    @pytest.mark.parametrize("gate", [None, 0.1])
    def test_gate_at_or_below_threshold_changes_nothing(self, gate):
        out = _cascade_engine()._run_cascade(
            "col", "stage2",
            [_strategy("value", [("sex", 0.25, "x")]),
             _strategy("ontology", [("disease", 0.9, "x")])],
            min_score=gate)
        assert out["method"] == "value"
