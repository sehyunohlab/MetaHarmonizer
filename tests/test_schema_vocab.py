"""schema_vocab + harmonize(schema_table=...) — OLS and engines are faked."""
import pandas as pd
import pytest

from metaharmonizer import pipeline, schema_vocab as sv
from metaharmonizer.pipeline import harmonize

DICT = pd.DataFrame({
    "ColName": ["sample_id", "disease", "sex", "hla", "age", "body_site_details", "smoker"],
    "ColClass": ["character", "character", "character", "character", "integer", "character", "character"],
    "MultipleValues": ["FALSE", "TRUE", "FALSE", "TRUE", "FALSE", "FALSE", "FALSE"],
    "AllowedValues": ["[0-9a-zA-Z]\\S+", "NA", "Female;Male",
                      "HLA-DQA1*02:01 protein complex;Novel HLA", "[0-9]+", "NA",
                      "smoker (finding);Non-smoker (finding)"],
    "DynamicEnum": ["NA", "NCIT:C7057;EFO:0000408", "NA", "MRO:0001676", "NA", "NA", "NA"],
    "DynamicEnumProperty": ["NA", "descendant", "NA", "descendant", "NA", "NA", "NA"],
})

TERMS = {
    ("NCIT:C7057", "descendant"): [
        {"label": "Crohn Disease", "obo_id": "NCIT:C2965"},
        {"label": "Healthy", "obo_id": "NCIT:C115935"},
        {"label": "imported", "obo_id": "MONDO:0000001"},   # other prefix: dropped
    ],
    ("EFO:0000408", "descendant"): [],                         # obsolete root
    ("MRO:0001676", "descendant"): [
        {"label": "HLA-DQA1*02:01 protein complex", "obo_id": "MRO:0000123"},
    ],
}


def fake_fetch(root, relation):
    return TERMS[(root, relation)]


def test_parse_allowed_values_patterns_and_labels():
    assert sv.parse_allowed_values("[0-9]+|NA") == ()
    assert sv.parse_allowed_values(".+") == ()
    assert sv.parse_allowed_values("^[1-9]\\d*(\\.\\d+)?$") == ()
    assert sv.parse_allowed_values("t[x1-4]n[x0-3]m[x0-1]|pTis|Tis") == ()
    assert sv.parse_allowed_values("NA") == ()
    # '*' and parentheses occur in real labels
    assert sv.parse_allowed_values("HLA-DQA1*02:01 protein complex;smoker (finding)") == (
        "HLA-DQA1*02:01 protein complex", "smoker (finding)")
    assert sv.parse_allowed_values("Yes;No;Yes") == ("Yes", "No")


def test_read_schema_table():
    v = sv.read_schema_table(DICT)
    assert v["disease"].roots == ("NCIT:C7057", "EFO:0000408")
    assert v["disease"].multiple_values and not v["sex"].multiple_values
    assert v["sex"].allowed_values == ("Female", "Male") and v["sex"].roots == ()
    assert not v["age"].has_vocabulary and not v["sample_id"].has_vocabulary
    assert not v["body_site_details"].has_vocabulary


def test_target_schema_frame_keeps_ncit_roots_only():
    ts = sv.target_schema_frame(DICT).set_index("field_name")
    assert ts.loc["disease", "ontology_root"] == "NCIT:C7057"
    assert ts.loc["hla", "ontology_root"] == ""
    assert ts.loc["age", "is_numeric_field"] == "yes"
    assert ts.loc["sex", "is_numeric_field"] == "no"


def test_build_corpora(tmp_path):
    fc = sv.build_field_corpora(sv.read_schema_table(DICT), tmp_path, fake_fetch)
    assert set(fc) == {"disease", "sex", "hla", "smoker"}
    (dis,) = fc["disease"]                       # EFO root obsolete -> no EFO corpus
    assert dis.ontology_source == "ncit" and not dis.local
    assert set(dis.corpus_df["obo_id"]) == {"NCIT:C2965", "NCIT:C115935"}
    (sex,) = fc["sex"]
    assert sex.local and list(sex.corpus_df["label"]) == ["Female", "Male"]
    assert all(c.startswith("LOCAL:") for c in sex.corpus_df["obo_id"])
    # hla: MRO corpus + a local corpus with only the label MRO lacks
    hla_ont, hla_loc = fc["hla"]
    assert hla_ont.ontology_source == "mro"
    assert list(hla_loc.corpus_df["label"]) == ["Novel HLA"]
    # root corpora are cached on disk and reused without fetching
    assert any(tmp_path.glob("ncit_c7057__descendant.csv"))
    again = sv.root_corpus("NCIT:C7057", "descendant", tmp_path,
                           fetch=lambda *_: pytest.fail("should use cache"))
    assert len(again) == 2


class FakeSM:
    def __init__(self, df, schema_map):
        self.df, self._schema_map = df, schema_map

    def run_schema_mapping(self):
        return self._schema_map


class FakeOM:
    calls = []

    def __init__(self, category, query_ls, **kw):
        FakeOM.calls.append((category, list(query_ls), kw))
        self.category, self.query_ls = category, list(query_ls)
        self.other_params = {"corpus_df": kw["corpus_df"].assign(
            official_label=kw["corpus_df"]["label"],
            clean_code=kw["corpus_df"]["obo_id"].str.replace("NCIT:", "", regex=False)
                                                .str.replace(":", "_", regex=False))}

    def run(self):
        labels = list(self.other_params["corpus_df"]["label"])
        rows = []
        for q in self.query_ls:
            hit = next((l for l in labels if l.lower() == q.lower()), labels[0])
            rows.append({"query": q, "match1": hit,
                         "match1_score": 1.0 if hit.lower() == q.lower() else 0.4})
        return pd.DataFrame(rows)


@pytest.fixture
def run(monkeypatch, tmp_path):
    FakeOM.calls = []
    data = pd.DataFrame({
        "sid": ["a", "b", "c"],
        "dx": ["crohn disease", "healthy", "healthy"],
        "gender": ["female", "male", "female"],
        "HLA": ["HLA-DQA1*02:01 protein complex", "Novel HLA", None],
    }, dtype=str)
    schema_map = pd.DataFrame([
        {"query": "sid", "stage": "stage1", "match1": "sample_id", "match1_score": 1.0},
        {"query": "dx", "stage": "stage1", "match1": "disease", "match1_score": 1.0},
        {"query": "gender", "stage": "stage3", "match1": "sex", "match1_score": 0.9},
        {"query": "HLA", "stage": "stage1", "match1": "hla", "match1_score": 1.0},
    ])
    seen = {}

    def make_sm(path, **kw):
        seen["sm_kwargs"] = kw
        return FakeSM(data, schema_map)

    monkeypatch.setattr(pipeline, "_make_schema_engine", make_sm)
    monkeypatch.setattr(pipeline, "_make_onto_engine", FakeOM)
    monkeypatch.setattr(sv, "_fetch_terms", fake_fetch)
    res = harmonize("x.csv", DICT, vocab_cache_dir=tmp_path,
                    id_columns=["sid"])
    return res, seen


def test_schema_table_drives_schema_and_routing(run):
    res, seen = run
    ts = pd.read_csv(seen["sm_kwargs"]["target_schema_path"], dtype=str)
    assert "smoker" in set(ts["field_name"])
    # sample_id has no vocabulary -> not routed; hla routed to two corpora
    assert set(res.routing["curated_field"]) == {"disease", "sex", "hla"}
    assert (res.routing["source_column"] == "HLA").sum() == 2


def test_best_match_across_corpora_and_local_term_ids(run):
    res, _ = run
    h = res.harmonized.set_index(["sid", "curated_field"])
    assert h.loc[("a", "disease"), "term_id"] == "NCIT:C2965"
    assert h.loc[("a", "sex"), "value"] == "Female"
    assert h.loc[("a", "sex"), "term_id"] == ""            # local label: no CURIE
    assert h.loc[("a", "hla"), "term_id"] == "MRO:0000123"  # ontology corpus wins
    assert h.loc[("b", "hla"), "value"] == "Novel HLA"      # local corpus wins
    assert h.loc[("b", "hla"), "ontology_source"] == "local"
    # one combined row per (column, value) after best-of
    assert not res.combined.duplicated(["source_column", "value"]).any()


def test_multi_valued_defaults_from_schema_table(monkeypatch, tmp_path):
    seen = {}
    real = pipeline._build_harmonized

    def spy(data, routing, combined, multi, noise):
        seen["multi"] = set(multi)
        return real(data, routing, combined, multi, noise)

    data = pd.DataFrame({"dx": ["healthy"]}, dtype=str)
    sm = pd.DataFrame([{"query": "dx", "stage": "stage1", "match1": "disease", "match1_score": 1.0}])
    monkeypatch.setattr(pipeline, "_make_schema_engine", lambda p, **k: FakeSM(data, sm))
    monkeypatch.setattr(pipeline, "_make_onto_engine", FakeOM)
    monkeypatch.setattr(pipeline, "_build_harmonized", spy)
    monkeypatch.setattr(sv, "_fetch_terms", fake_fetch)
    harmonize("x.csv", DICT, vocab_cache_dir=tmp_path)
    assert seen["multi"] == {"disease", "hla"}


def test_wide_follows_schema_table_field_order(run):
    res, _ = run
    w = res.wide()
    fields = [c for c in w.columns if c in set(DICT["ColName"])]
    assert fields == [f for f in DICT["ColName"] if f in fields]
    # local-only vocabulary (sex): labels, no term-ID column
    assert "sex" in w.columns and "sex_ontology_term_id" not in w.columns
    assert "disease_ontology_term_id" in w.columns


# --- drafting a schema table from a preset ------------------------------------

def test_draft_schema_table_from_cbio_preset():
    d = sv.draft_schema_table("cbio")
    assert list(d.columns) == sv.SCHEMA_TABLE_COLUMNS
    from metaharmonizer.models.schema_mapper.config import resolve_schema_preset
    fields = pd.read_csv(resolve_schema_preset("cbio")["target_schema_path"])["field_name"]
    assert list(d["ColName"]) == list(fields)
    r = d.set_index("ColName")
    assert r.loc["cancer_type", "DynamicEnum"] == "NCIT:C3262"
    assert r.loc["cancer_type", "DynamicEnumProperty"] == "descendant"
    assert r.loc["body_site", "DynamicEnum"] == "NCIT:C32221"
    assert r.loc["sex", "AllowedValues"] and r.loc["sex", "DynamicEnum"] == ""
    assert r.loc["age_at_diagnosis", "ColClass"] == "numeric"
    # round-trips: read_schema_table sees the same vocabularies
    v = sv.read_schema_table(d)
    assert v["cancer_type"].roots == ("NCIT:C3262",)
    assert v["sex"].allowed_values == tuple(r.loc["sex", "AllowedValues"].split(";"))
    assert sv.draft_schema_table("cbio", allowed_values=False)["AllowedValues"].eq("").all()


def test_draft_skips_allowed_values_that_cannot_round_trip():
    r = sv.draft_schema_table("gdc").set_index("ColName")
    # GDC labels such as "Colonization; bacterial" contain the separator
    assert r.loc["additional_pathology_findings", "AllowedValues"] == ""


def test_preset_roots_are_registry_corpora():
    from metaharmonizer.engine.ontology_mapping_engine import _CORPUS_REGISTRY
    for fields in sv.PRESET_FIELD_CORPORA.values():
        assert all(spec in _CORPUS_REGISTRY for spec in fields.values())


def test_write_schema_table_draft_copies_preset_dictionaries(tmp_path):
    p = sv.write_schema_table_draft("cbio", tmp_path / "cbio_schema.csv")
    assert (tmp_path / "cbio_schema.alias.csv").exists()
    assert (tmp_path / "cbio_schema.values.json").exists()
    assert pd.read_csv(p, dtype=str, keep_default_na=False).equals(sv.draft_schema_table("cbio"))
    with pytest.raises(FileExistsError):
        sv.write_schema_table_draft("cbio", p)
    sv.write_schema_table_draft("cbio", p, overwrite=True)
