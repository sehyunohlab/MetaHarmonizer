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


# --- cBioPortal-style dictionary (dotted lower-case headers, static enums) ----

CBIO_DICT = pd.DataFrame({
    "col.name": ["age_group", "sex", "age_at_death", "cancer_type"],
    "col.class": ["character", "character", "numeric", "character"],
    "multiplevalues": ["FALSE", "FALSE", "FALSE", "TRUE"],
    "allowedvalues": ["Infant|Adult|Elderly", "Female;Male", "[0-9]+", "NA"],
    "dynamic.enum": ["NA", "NA", "NA", "NCIT:C3262"],
    "dynamic.enum.property": ["NA", "NA", "NA", "descendant"],
    "corpus.type": ["static_enum", "NA", "regexp", "dynamic_enum"],
    "static.enum": ["NCIT:C27956|NCIT:C17600|NCIT:C16268", "NA", "NA", "NA"],
})


def test_reads_dotted_headers_and_static_enum_codes():
    v = sv.read_schema_table(CBIO_DICT)
    assert v["age_group"].allowed_values == ("Infant", "Adult", "Elderly")
    assert v["age_group"].allowed_codes == ("NCIT:C27956", "NCIT:C17600", "NCIT:C16268")
    assert v["sex"].allowed_values == ("Female", "Male") and v["sex"].allowed_codes == ("", "")
    assert not v["age_at_death"].has_vocabulary
    assert v["cancer_type"].roots == ("NCIT:C3262",) and v["cancer_type"].multiple_values
    ts = sv.target_schema_frame(CBIO_DICT).set_index("field_name")
    assert ts.loc["age_at_death", "is_numeric_field"] == "yes"
    assert ts.loc["cancer_type", "ontology_root"] == "NCIT:C3262"


def test_static_enum_with_mismatched_codes_keeps_labels_only():
    d = CBIO_DICT.copy()
    d.loc[0, "static.enum"] = "NCIT:C27956"
    v = sv.read_schema_table(d)["age_group"]
    assert len(v.allowed_values) == 3 and v.allowed_codes == ("", "", "")


def test_static_enum_codes_become_term_ids(monkeypatch, tmp_path):
    (c,) = sv.build_corpora(sv.read_schema_table(CBIO_DICT)["age_group"], tmp_path)
    assert c.local and c.corpus_df["obo_id"].str.startswith("LOCAL:").all()
    assert pipeline._label_to_term_id(c) == {"infant": "NCIT:C27956", "adult": "NCIT:C17600",
                                             "elderly": "NCIT:C16268"}
    (s,) = sv.build_corpora(sv.read_schema_table(CBIO_DICT)["sex"], tmp_path)
    assert pipeline._label_to_term_id(s) == {}


# --- allowed values whose label is also a dynamic term (PR #108 review) ------

CONFLICT_DICT = pd.DataFrame({
    "col.name": ["status"],
    "allowedvalues": ["Healthy|Crohn Disease|Remission"],
    "dynamic.enum": ["NCIT:C7057"],
    "dynamic.enum.property": ["descendant"],
    "corpus.type": ["static_enum"],
    # Healthy: different ID than the dynamic term; Crohn Disease: same ID;
    # Remission: not in the dynamic corpus
    "static.enum": ["NCIT:C999999|NCIT:C2965|NCIT:C4870"],
})


def test_allowed_value_with_a_different_static_id_keeps_both_entries(tmp_path):
    ont, loc = sv.build_corpora(sv.read_schema_table(CONFLICT_DICT)["status"],
                                tmp_path, fake_fetch)
    assert "Healthy" in set(ont.corpus_df["label"])            # dynamic entry kept
    assert list(loc.corpus_df["label"]) == ["Healthy", "Remission"]
    assert loc.term_ids == {"healthy": "NCIT:C999999", "remission": "NCIT:C4870"}


def test_allowed_value_without_static_id_is_still_covered_by_dynamic_label(tmp_path):
    _, loc = sv.build_field_corpora(sv.read_schema_table(DICT), tmp_path, fake_fetch)["hla"]
    assert list(loc.corpus_df["label"]) == ["Novel HLA"]


def test_static_id_wins_a_tie_with_the_dynamic_term(monkeypatch, tmp_path):
    data = pd.DataFrame({"st": ["healthy", "crohn disease"]}, dtype=str)
    sm = pd.DataFrame([{"query": "st", "stage": "stage1", "match1": "status",
                        "match1_score": 1.0}])
    monkeypatch.setattr(pipeline, "_make_schema_engine", lambda p, **k: FakeSM(data, sm))
    monkeypatch.setattr(pipeline, "_make_onto_engine", FakeOM)
    monkeypatch.setattr(sv, "_fetch_terms", fake_fetch)
    res = harmonize("x.csv", CONFLICT_DICT, vocab_cache_dir=tmp_path)
    c = res.combined.set_index("value")
    assert c.loc["healthy", "match1_term_id"] == "NCIT:C999999"
    assert c.loc["healthy", "ontology_source"] == "local"
    assert c.loc["crohn disease", "match1_term_id"] == "NCIT:C2965"
