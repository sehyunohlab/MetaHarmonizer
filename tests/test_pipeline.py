"""Tests for metaharmonizer.pipeline.harmonize — OLS and engines are faked, no models load."""
import pandas as pd
import pytest

from metaharmonizer import pipeline, schema_vocab as sv
from metaharmonizer.pipeline import ROW_ID, harmonize


def schema(**roots):
    """A schema table: field -> DynamicEnum root ('' for no vocabulary)."""
    return pd.DataFrame({"ColName": list(roots), "DynamicEnum": list(roots.values()),
                         "DynamicEnumProperty": ["descendant" if r else "" for r in roots.values()]})


def fake_fetch(table):
    def fetch(root, relation):
        return [{"label": lab, "obo_id": code} for lab, code in table[root]]
    return fetch


class FakeSM:
    def __init__(self, df, schema_map):
        self.df = df
        self._schema_map = schema_map

    def run_schema_mapping(self):
        return self._schema_map


class FakeOM:
    """Returns the corpus label equal to the query, else the first label."""
    calls = []

    def __init__(self, category, query_ls, **kwargs):
        FakeOM.calls.append((category, list(query_ls), kwargs))
        self.query_ls = list(query_ls)
        self.labels = list(kwargs["corpus_df"]["label"])
        if kwargs.get("fail"):
            raise RuntimeError("corpus unavailable")

    def run(self):
        hit = [next((l for l in self.labels if l.lower() == q.lower()), self.labels[0])
               for q in self.query_ls]
        return pd.DataFrame({"query": self.query_ls, "match1": hit,
                             "match1_score": [0.9] * len(hit), "stage": [2.0] * len(hit)})


CBIO_TERMS = {
    "NCIT:C3262": [("Lung Adenocarcinoma", "NCIT:C3512"), ("Melanoma", "NCIT:C3224"),
                   ("Breast Carcinoma", "NCIT:C4872")],
    "NCIT:C32221": [("Lung", "NCIT:C12468"), ("Skin", "NCIT:C12470")],
    "NCIT:C1909": [("Cisplatin", "NCIT:C376")],
}
CBIO = schema(disease="NCIT:C3262", cancer_type="NCIT:C3262", body_site="NCIT:C32221",
              treatment_name="NCIT:C1909", age_at_diagnosis="")
DISEASE, BODYSITE = "v_ncit_c3262", "v_ncit_c32221"


@pytest.fixture
def patched(monkeypatch, tmp_path):
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
    monkeypatch.setattr(sv, "_fetch_terms", fake_fetch(CBIO_TERMS))
    seen["run"] = lambda table=CBIO, **kw: harmonize("x.csv", table, vocab_cache_dir=tmp_path, **kw)
    return seen


def test_routing_gates_on_vocabulary_and_score(patched):
    res = patched["run"]()
    assert list(res.routing["source_column"]) == ["Diagnosis", "Tumor Type", "Site"]
    # Drug: score 0.3 < 0.8; Age: field has no vocabulary (so not in skipped either)
    assert "Drug" not in set(res.combined["source_column"])
    assert list(res.skipped["source_column"]) == ["Drug"]
    assert res.skipped["reason"].iloc[0].startswith("schema score")
    # the schema table is the SchemaMapper target schema; no preset is selected
    assert patched["sm_kwargs"]["schema"] is None
    ts = pd.read_csv(patched["sm_kwargs"]["target_schema_path"])
    assert list(ts["field_name"]) == list(CBIO["ColName"])


def test_fields_sharing_a_corpus_are_pooled_into_one_run(patched):
    res = patched["run"]()
    cats = [c[0] for c in FakeOM.calls]
    assert sorted(cats) == sorted([DISEASE, BODYSITE])  # one run each
    disease_vals = next(q for c, q, _ in FakeOM.calls if c == DISEASE)
    # deduped across Diagnosis + Tumor Type, stripped, noise/empty dropped
    assert disease_vals == ["lung adenocarcinoma", "melanoma", "breast carcinoma"]
    assert set(res.ontology_maps) == {(DISEASE, "ncit"), (BODYSITE, "ncit")}
    for _, _, kw in FakeOM.calls:
        assert kw["ontology_source"] == "ncit"
        assert kw["corpus_df"]["obo_id"].str.startswith("NCIT:").all()


def test_combined_has_one_row_per_column_value(patched):
    c = patched["run"]().combined
    diag = c[c["source_column"] == "Diagnosis"]
    assert list(diag["value"]) == ["lung adenocarcinoma", "melanoma"]
    assert list(diag["match1"]) == ["Lung Adenocarcinoma", "Melanoma"]
    assert list(diag["match1_term_id"]) == ["NCIT:C3512", "NCIT:C3224"]
    tumor = c[c["source_column"] == "Tumor Type"]
    assert list(tumor["value"]) == ["melanoma", "breast carcinoma"]
    assert (tumor["curated_field"] == "cancer_type").all()
    assert list(c.columns[:7]) == [
        "source_column", "curated_field", "schema_score", "schema_stage",
        "corpus_category", "ontology_source", "value"]


def test_numeric_and_identifier_columns_are_skipped(monkeypatch, tmp_path):
    FakeOM.calls = []
    n = 30
    data = pd.DataFrame({
        "patientId": [f"TCGA-{i:04d}" for i in range(n)],
        "MUTATION_COUNT": [str(i) for i in range(n)],
        "Diagnosis": ["melanoma", "carcinoma", "lymphoma"] * 10,
    }, dtype=str)
    schema_map = pd.DataFrame([
        {"query": "patientId", "stage": "stage3", "match1": "body_site", "match1_score": 0.9},
        {"query": "MUTATION_COUNT", "stage": "stage3", "match1": "cancer_type", "match1_score": 0.9},
        {"query": "Diagnosis", "stage": "stage1", "match1": "disease", "match1_score": 1.0},
    ])
    monkeypatch.setattr(pipeline, "_make_schema_engine",
                        lambda path, **kw: FakeSM(data, schema_map))
    monkeypatch.setattr(pipeline, "_make_onto_engine", FakeOM)
    monkeypatch.setattr(sv, "_fetch_terms", fake_fetch(CBIO_TERMS))

    res = harmonize("x.csv", CBIO, vocab_cache_dir=tmp_path)
    assert list(res.routing["source_column"]) == ["Diagnosis"]
    reasons = dict(zip(res.skipped["source_column"], res.skipped["reason"]))
    assert reasons["patientId"].startswith("identifier-like")
    assert reasons["MUTATION_COUNT"].startswith("numeric")

    FakeOM.calls = []
    res = harmonize("x.csv", CBIO, vocab_cache_dir=tmp_path,
                    max_unique_ratio=None, max_numeric_frac=1.0)
    assert list(res.routing["source_column"]) == ["patientId", "MUTATION_COUNT", "Diagnosis"]


def test_keep_noise_values_when_disabled(patched):
    patched["run"](drop_noise_values=False)
    disease_vals = next(q for c, q, _ in FakeOM.calls if c == DISEASE)
    assert "Unknown" in disease_vals


def test_threshold(patched):
    res = patched["run"](schema(treatment_name="NCIT:C1909"), min_schema_score=0.0)
    assert list(res.routing["source_column"]) == ["Drug"]
    assert FakeOM.calls[0][1] == ["cisplatin", "x", "y"]


def test_onto_kwargs_forwarded(patched):
    patched["run"](onto_kwargs={"top_k": 3, "s3_strategy": "rag"})
    assert all(kw["top_k"] == 3 and kw["s3_strategy"] == "rag" for _, _, kw in FakeOM.calls)


@pytest.mark.parametrize("key", ["corpus_df", "ontology_source"])
def test_onto_kwargs_cannot_override_the_schema_corpus(patched, key):
    with pytest.raises(ValueError, match="come from schema_table"):
        patched["run"](onto_kwargs={key: "x"})


def test_om_failure_is_isolated(patched):
    res = patched["run"](onto_kwargs={"fail": True})
    assert set(res.errors) == {(DISEASE, "ncit"), (BODYSITE, "ncit")}
    assert "corpus unavailable" in res.errors[(DISEASE, "ncit")]
    assert res.combined.empty
    with pytest.raises(RuntimeError):
        patched["run"](onto_kwargs={"fail": True}, raise_on_error=True)


@pytest.mark.parametrize("table", [None, "cbio", "gdc"])
def test_schema_table_is_required(patched, table):
    with pytest.raises(ValueError, match="write_schema_table_draft"):
        harmonize("x.csv", table)


def test_sibling_dictionaries_go_to_schema_mapper(patched, tmp_path):
    path = tmp_path / "my_schema.csv"
    CBIO.to_csv(path, index=False)
    (tmp_path / "my_schema.alias.csv").write_text("field_name,source\n")
    patched["run"](str(path))
    kw = patched["sm_kwargs"]
    assert kw["alias_dict_path"] == str(tmp_path / "my_schema.alias.csv")
    assert "value_dict_path" not in kw          # no sibling -> engine default


def test_lazy_top_level_export():
    import metaharmonizer
    assert metaharmonizer.harmonize is harmonize


# ---------------------------------------------------------------------------
# Per-sample merge (HarmonizeResult.harmonized / .wide)
# ---------------------------------------------------------------------------

UBERON = {"stool": ("feces", "UBERON:0001988"), "oralcavity": ("mouth", "UBERON:0000165"),
          "rectum": ("rectum", "UBERON:0001052"), "left_colon": ("left colon", "UBERON:0008971"),
          "hpc": ("preculminate fissure", "UBERON:0035926")}
NCIT = {"T2D": ("Type 2 Diabetes Mellitus", "C26747"), "healthy": ("Healthy", "C115935"),
        "type 2 diabetes": ("Type 2 Diabetes Mellitus", "C26747"),
        "schizophrenia": ("Schizophrenia", "C3362")}


class CorpusOM:
    """OM double that maps via a fixed table."""

    def __init__(self, category, query_ls, **kwargs):
        self.table = UBERON if kwargs["ontology_source"] == "uberon" else NCIT
        self.query_ls = list(query_ls)

    def run(self):
        hit = [self.table.get(q, ("Unrelated Thing", "X")) for q in self.query_ls]
        return pd.DataFrame({"query": self.query_ls, "match1": [h[0] for h in hit],
                             "match1_score": [0.9] * len(self.query_ls), "stage": [2.0] * len(self.query_ls)})


@pytest.fixture
def merge_case(monkeypatch, tmp_path):
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
    terms = {"UBERON:0001062": list(UBERON.values()),
             "NCIT:C7057": [(lab, "NCIT:" + c) for lab, c in NCIT.values()]}
    monkeypatch.setattr(sv, "_fetch_terms", fake_fetch(terms))
    table = schema(sample_id="", body_site="UBERON:0001062", disease="NCIT:C7057", age="")
    return lambda **kw: harmonize("x.csv", table, vocab_cache_dir=tmp_path,
                                  min_schema_score=0.6, max_unique_ratio=None,
                                  **{"id_columns": ["sample_id"], **kw})


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
    assert list(w.columns) == [ROW_ID, "sample_id", "body_site", "body_site_ontology_term_id",
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
    with pytest.raises(ValueError, match="id_columns not in input"):
        merge_case(id_columns=["nope"])


@pytest.mark.parametrize("col", [ROW_ID, "value", "term_id", "disease"])
def test_id_columns_must_not_clash_with_generated_columns(merge_case, col):
    with pytest.raises(ValueError, match="clash"):
        merge_case(id_columns=[col])


def test_input_column_named_row_is_carried_as_an_id(merge_case, monkeypatch):
    real = pipeline._make_schema_engine

    def with_row(path, **kw):
        sm = real(path, **kw)
        sm.df = sm.df.assign(row=[f"r{i}" for i in range(len(sm.df))])
        return sm

    monkeypatch.setattr(pipeline, "_make_schema_engine", with_row)
    res = merge_case(id_columns=["row"])
    assert list(res.harmonized.columns[:2]) == [ROW_ID, "row"]
    assert list(res.wide()["row"]) == ["r0", "r1", "r2", "r3", "r4"]


def test_term_ids_are_the_corpus_curies(merge_case):
    h = merge_case().harmonized
    assert set(h["term_id"]) <= {"UBERON:0001988", "UBERON:0000165", "UBERON:0001052",
                                 "NCIT:C26747", "NCIT:C115935", "NCIT:C3362"}
    assert set(h["ontology_source"]) == {"uberon", "ncit"}
