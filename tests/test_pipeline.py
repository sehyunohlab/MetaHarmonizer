"""Tests for metaharmonizer.pipeline.harmonize — engines are faked, no models load."""
import pandas as pd
import pytest

from metaharmonizer import pipeline
from metaharmonizer.pipeline import FIELD_CATEGORY_PRESETS, harmonize


class FakeSM:
    def __init__(self, df, schema_map):
        self.df = df
        self._schema_map = schema_map

    def run_schema_mapping(self):
        return self._schema_map


class FakeOM:
    calls = []

    def __init__(self, category, query_ls, **kwargs):
        FakeOM.calls.append((category, list(query_ls), kwargs))
        self.category = category
        self.query_ls = list(query_ls)
        if kwargs.get("fail"):
            raise RuntimeError("corpus unavailable")

    def run(self):
        return pd.DataFrame({
            "query": self.query_ls,
            "match1": [f"{self.category}:{q.upper()}" for q in self.query_ls],
            "match1_score": [0.9] * len(self.query_ls),
            "stage": [2.0] * len(self.query_ls),
        })


@pytest.fixture
def patched(monkeypatch):
    FakeOM.calls = []
    data = pd.DataFrame({
        "Diagnosis": ["lung adenocarcinoma", "Unknown", None, " melanoma "],
        "Tumor Type": ["melanoma", "breast carcinoma", "melanoma", ""],
        "Site": ["lung", "skin", "lung", "NA"],
        "Drug": ["cisplatin", "cisplatin", "x", "y"],
        "Age": ["50", "60", "70", "80"],
    }, dtype=str)
    schema_map = pd.DataFrame([
        {"query": "Diagnosis", "stage": "stage1", "match1": "disease", "match1_score": 1.0},
        {"query": "Tumor Type", "stage": "stage3", "match1": "cancer_type", "match1_score": 0.85},
        {"query": "Site", "stage": "stage2", "match1": "body_site", "match1_score": 0.9},
        {"query": "Drug", "stage": "stage3", "match1": "treatment_name", "match1_score": 0.3},
        {"query": "Age", "stage": "stage1", "match1": "age_at_diagnosis", "match1_score": 1.0},
    ])
    seen = {}

    def make_sm(input_path, **kwargs):
        seen["sm_kwargs"] = kwargs
        return FakeSM(data, schema_map)

    monkeypatch.setattr(pipeline, "_make_schema_engine", make_sm)
    monkeypatch.setattr(pipeline, "_make_onto_engine", FakeOM)
    return seen


def test_routing_gates_on_preset_and_score(patched):
    res = harmonize("x.csv", schema="cbio")
    assert list(res.routing["source_column"]) == ["Diagnosis", "Tumor Type", "Site"]
    # Drug: score 0.3 < 0.8; Age: not ontology-backed (so not in skipped either)
    assert "Drug" not in set(res.combined["source_column"])
    assert list(res.skipped["source_column"]) == ["Drug"]
    assert res.skipped["reason"].iloc[0].startswith("schema score")
    assert patched["sm_kwargs"]["schema"] == "cbio"


def test_columns_sharing_a_category_are_pooled_into_one_run(patched):
    res = harmonize("x.csv", schema="cbio")
    cats = [c[0] for c in FakeOM.calls]
    assert sorted(cats) == ["bodysite", "disease"]  # one run each
    disease_vals = next(q for c, q, _ in FakeOM.calls if c == "disease")
    # deduped across Diagnosis + Tumor Type, stripped, noise/empty dropped
    assert disease_vals == ["lung adenocarcinoma", "melanoma", "breast carcinoma"]
    assert set(res.ontology_maps) == {("disease", "ncit"), ("bodysite", "ncit")}
    assert all(kw["ontology_source"] == "ncit" for _, _, kw in FakeOM.calls)


def test_combined_has_one_row_per_column_value(patched):
    res = harmonize("x.csv", schema="cbio")
    c = res.combined
    diag = c[c["source_column"] == "Diagnosis"]
    assert list(diag["value"]) == ["lung adenocarcinoma", "melanoma"]
    assert list(diag["match1"]) == ["disease:LUNG ADENOCARCINOMA", "disease:MELANOMA"]
    tumor = c[c["source_column"] == "Tumor Type"]
    assert list(tumor["value"]) == ["melanoma", "breast carcinoma"]
    assert (tumor["curated_field"] == "cancer_type").all()
    assert list(c.columns[:7]) == [
        "source_column", "curated_field", "schema_score", "schema_stage",
        "corpus_category", "ontology_source", "value"]


def test_numeric_and_identifier_columns_are_skipped(monkeypatch):
    FakeOM.calls = []
    n = 30
    data = pd.DataFrame({
        "patientId": [f"TCGA-{i:04d}" for i in range(n)],
        "MUTATION_COUNT": [str(i) for i in range(n)],
        "Diagnosis": ["melanoma", "carcinoma", "lymphoma"] * 10,
    }, dtype=str)
    schema_map = pd.DataFrame([
        {"query": "patientId", "stage": "stage3", "match1": "body_site", "match1_score": 0.9},
        {"query": "MUTATION_COUNT", "stage": "stage3", "match1": "cancer_subtype", "match1_score": 0.9},
        {"query": "Diagnosis", "stage": "stage1", "match1": "disease", "match1_score": 1.0},
    ])
    monkeypatch.setattr(pipeline, "_make_schema_engine",
                        lambda path, **kw: FakeSM(data, schema_map))
    monkeypatch.setattr(pipeline, "_make_onto_engine", FakeOM)

    res = harmonize("x.csv", schema="cbio")
    assert list(res.routing["source_column"]) == ["Diagnosis"]
    reasons = dict(zip(res.skipped["source_column"], res.skipped["reason"]))
    assert reasons["patientId"].startswith("identifier-like")
    assert reasons["MUTATION_COUNT"].startswith("numeric")

    FakeOM.calls = []
    res = harmonize("x.csv", schema="cbio", max_unique_ratio=None, max_numeric_frac=1.0)
    assert list(res.routing["source_column"]) == ["patientId", "MUTATION_COUNT", "Diagnosis"]


def test_keep_noise_values_when_disabled(patched):
    harmonize("x.csv", schema="cbio", drop_noise_values=False)
    disease_vals = next(q for c, q, _ in FakeOM.calls if c == "disease")
    assert "Unknown" in disease_vals


def test_custom_field_categories_and_threshold(patched):
    res = harmonize("x.csv", schema="cbio", min_schema_score=0.0,
                    field_categories={"treatment_name": ("treatment", "ncit")})
    assert list(res.routing["source_column"]) == ["Drug"]
    assert FakeOM.calls[0][1] == ["cisplatin", "x", "y"]


def test_onto_kwargs_forwarded(patched):
    harmonize("x.csv", schema="cbio", onto_kwargs={"top_k": 3, "s3_strategy": "rag"})
    assert all(kw["top_k"] == 3 and kw["s3_strategy"] == "rag" for _, _, kw in FakeOM.calls)


def test_om_failure_is_isolated(patched):
    res = harmonize("x.csv", schema="cbio", onto_kwargs={"fail": True})
    assert set(res.errors) == {("disease", "ncit"), ("bodysite", "ncit")}
    assert "corpus unavailable" in res.errors[("disease", "ncit")]
    assert res.combined.empty
    with pytest.raises(RuntimeError):
        harmonize("x.csv", schema="cbio", onto_kwargs={"fail": True}, raise_on_error=True)


def test_no_preset_requires_field_categories(patched):
    with pytest.raises(ValueError, match="field_categories"):
        harmonize("x.csv", schema=None)
    with pytest.raises(ValueError, match="custom target_schema_path"):
        harmonize("x.csv", schema="cbio", schema_kwargs={"target_schema_path": "s.csv"})


def test_presets_reference_real_fields_and_registry_pairs():
    from metaharmonizer.engine.ontology_mapping_engine import _CORPUS_REGISTRY
    from metaharmonizer.models.schema_mapper.config import resolve_schema_preset

    for schema, fields in FIELD_CATEGORY_PRESETS.items():
        schema_fields = set(
            pd.read_csv(resolve_schema_preset(schema)["target_schema_path"])["field_name"])
        for fld, spec in fields.items():
            assert fld in schema_fields, f"{schema}: {fld} not in schema CSV"
            assert spec in _CORPUS_REGISTRY, f"{schema}: {spec} not in registry"


def test_lazy_top_level_export():
    import metaharmonizer
    assert metaharmonizer.harmonize is harmonize
