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


# ---------------------------------------------------------------------------
# Per-sample merge (HarmonizeResult.harmonized / .wide)
# ---------------------------------------------------------------------------

UBERON = {"stool": ("feces", "UBERON_0001988"), "oralcavity": ("mouth", "UBERON_0000165"),
          "rectum": ("rectum", "UBERON_0001052"), "left_colon": ("left colon", "UBERON_0008971"),
          "hpc": ("preculminate fissure", "UBERON_0035926")}
NCIT = {"T2D": ("Type 2 Diabetes Mellitus", "C26747"), "healthy": ("Healthy", "C115935"),
        "type 2 diabetes": ("Type 2 Diabetes Mellitus", "C26747"),
        "schizophrenia": ("Schizophrenia", "C3362")}


class CorpusOM:
    """OM double that maps via a fixed table and exposes a normalized corpus."""
    calls = []

    def __init__(self, category, query_ls, **kwargs):
        CorpusOM.calls.append(category)
        self.table = UBERON if category == "bodysite" else NCIT
        self.query_ls = list(query_ls)
        terms = {lab: code for lab, code in self.table.values()}
        self.other_params = {"corpus_df": pd.DataFrame(
            {"official_label": list(terms), "clean_code": list(terms.values())})}

    def run(self):
        hit = [self.table.get(q, ("Unrelated Thing", "X")) for q in self.query_ls]
        return pd.DataFrame({"query": self.query_ls, "match1": [h[0] for h in hit],
                             "match1_score": [0.9] * len(self.query_ls), "stage": [2.0] * len(self.query_ls)})


@pytest.fixture
def merge_case(monkeypatch):
    data = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        "body_site": ["stool", "stool", "stool", None, "oralcavity"],
        "disease_location": ["rectum", None, "left_colon", "rectum", None],
        "hitchip_probe_class": [None, "hpc", None, None, None],
        "disease": ["T2D", "healthy", "T2D", "unknown", "healthy"],
        "study_condition": ["type 2 diabetes", None, "schizophrenia", None, "healthy"],
        "subject_age": ["30", " 41 ", None, "", "nan"],
        "age_alt": ["99", None, "52", None, None],
        "age_noise": ["x"] * 5,
    }, dtype=object)
    schema_map = pd.DataFrame([
        {"query": "sample_id", "stage": "stage1", "match1": "sample_id", "match1_score": 1.0},
        {"query": "body_site", "stage": "stage1", "match1": "body_site", "match1_score": 1.0},
        {"query": "disease_location", "stage": "stage3", "match1": "body_site", "match1_score": 0.75},
        {"query": "hitchip_probe_class", "stage": "stage3", "match1": "body_site", "match1_score": 0.65},
        {"query": "disease", "stage": "stage1", "match1": "disease", "match1_score": 1.0},
        {"query": "study_condition", "stage": "stage3", "match1": "disease", "match1_score": 0.7},
        {"query": "subject_age", "stage": "stage3", "match1": "age", "match1_score": 0.9},
        {"query": "age_alt", "stage": "stage3", "match1": "age", "match1_score": 0.7},
        {"query": "age_noise", "stage": "stage3", "match1": "age", "match1_score": 0.5},
    ])
    monkeypatch.setattr(pipeline, "_make_schema_engine",
                        lambda path, **kw: FakeSM(data, schema_map))
    monkeypatch.setattr(pipeline, "_make_onto_engine", CorpusOM)
    cats = {"body_site": ("bodysite", "uberon"), "disease": ("disease", "ncit")}
    return lambda **kw: harmonize("x.csv", schema=None, field_categories=cats,
                                  min_schema_score=0.6, max_unique_ratio=None,
                                  id_columns=["sample_id"], **kw)


def _h(res, field):
    return res.harmonized[res.harmonized.curated_field == field].set_index("sample_id")


def test_single_valued_keeps_best_column_and_flags_the_rest(merge_case):
    bs = _h(merge_case(), "body_site")
    assert bs.loc["s1", "value"] == "feces"
    assert bs.loc["s1", "term_id"] == "UBERON:0001988"
    assert bs.loc["s1", "source_columns"] == "body_site"
    assert bs.loc["s1", "dropped"] == "disease_location=rectum→rectum"
    assert bs.loc["s1", "conflict"]
    assert bs.loc["s2", "dropped"] == "hitchip_probe_class=hpc→preculminate fissure"
    assert bs.loc["s5", "value"] == "mouth" and not bs.loc["s5", "conflict"]


def test_single_valued_falls_back_per_row_when_best_column_is_empty(merge_case):
    bs = _h(merge_case(), "body_site")
    # s4 has no body_site, so the next-best column that has a value is used
    assert bs.loc["s4", "value"] == "rectum"
    assert bs.loc["s4", "source_columns"] == "disease_location"
    assert bs.loc["s4", "dropped"] == "" and not bs.loc["s4", "conflict"]


def test_multi_valued_unions_and_dedupes_by_term(merge_case):
    d = _h(merge_case(multi_valued_fields={"disease"}), "disease")
    assert d.loc["s1", "value"] == "Type 2 Diabetes Mellitus"          # same term twice -> once
    assert d.loc["s1", "original_value"] == "T2D"
    assert d.loc["s3", "value"] == "Type 2 Diabetes Mellitus;Schizophrenia"
    assert d.loc["s3", "term_id"] == "NCIT:C26747;NCIT:C3362"
    assert d.loc["s3", "source_columns"] == "disease;study_condition"
    assert "s4" not in d.index                                           # 'unknown' is noise
    assert d["dropped"].eq("").all()


def test_same_term_from_dropped_column_is_not_a_conflict(merge_case):
    d = _h(merge_case(), "disease")  # disease single-valued here
    assert d.loc["s1", "dropped"] == "study_condition=type 2 diabetes→Type 2 Diabetes Mellitus"
    assert not d.loc["s1", "conflict"]
    assert d.loc["s3", "conflict"]                                       # Schizophrenia dropped


def test_wide_matches_release_layout(merge_case):
    w = merge_case(multi_valued_fields={"disease"}).wide()
    assert list(w.columns) == ["row", "sample_id", "body_site", "body_site_ontology_term_id",
                               "disease", "disease_ontology_term_id", "age"]
    assert len(w) == 5
    assert w.loc[2, "disease_ontology_term_id"] == "NCIT:C26747;NCIT:C3362"
    assert pd.isna(w.loc[3, "disease"])


def test_wide_passes_through_unmapped_fields(merge_case):
    res = merge_case()
    assert set(res.passthrough_routing["source_column"]) == {"subject_age", "age_alt"}  # 0.5 < 0.6
    w = res.wide().set_index("sample_id")
    # best-scoring column per row, falling back to the next one; blanks are NaN
    assert w.loc["s1", "age"] == "30" and w.loc["s2", "age"] == "41"
    assert w.loc["s3", "age"] == "52"
    assert pd.isna(w.loc["s4", "age"]) and pd.isna(w.loc["s5", "age"])
    assert "age" not in res.wide(passthrough=False).columns


def test_wide_provenance_columns(merge_case):
    w = merge_case().wide(provenance=True).set_index("sample_id")
    assert w.loc["s1", "body_site_source_column"] == "body_site"
    assert w.loc["s1", "body_site_original_value"] == "stool"
    assert w.loc["s1", "body_site_schema_score"] == 1.0
    assert w.loc["s1", "body_site_map_score"] == 0.9
    assert w.loc["s1", "body_site_dropped"] == "disease_location=rectum→rectum"
    assert w.loc["s4", "body_site_source_column"] == "disease_location"
    assert w.loc["s4", "body_site_schema_score"] == 0.75
    assert pd.isna(w.loc["s4", "body_site_dropped"])
    assert w.loc["s3", "age_source_column"] == "age_alt"
    assert w.loc["s3", "age_schema_score"] == 0.7
    assert "age_map_score" not in w.columns            # passthrough: no ontology mapping


def test_id_columns_validated(merge_case):
    with pytest.raises(ValueError, match="id_columns"):
        harmonize(
            "x.csv", schema=None, field_categories={"body_site": ("bodysite", "uberon")},
            id_columns=["nope"])


def test_to_curie():
    from metaharmonizer.pipeline import _to_curie
    assert _to_curie("C1909", "ncit") == "NCIT:C1909"
    assert _to_curie("UBERON_0001988", "uberon") == "UBERON:0001988"
    assert _to_curie("NCIT:C1909", "ncit") == "NCIT:C1909"
    assert _to_curie("local42", "uberon") == "local42"
    assert _to_curie("nan", "ncit") == ""


def test_harmonized_has_no_term_ids_when_engine_exposes_no_corpus(patched):
    res = harmonize("x.csv", schema="cbio")
    assert set(res.harmonized["curated_field"]) == {"disease", "cancer_type", "body_site"}
    assert res.harmonized["term_id"].eq("").all()   # FakeOM exposes no corpus
