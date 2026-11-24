"""schema_cells: one reading of a row's value columns for schema_vocab and
CombinedCorpusBuilder (MetaHarmonizerSchemaRegistry convention)."""
import pandas as pd

from metaharmonizer import schema_vocab as sv
from metaharmonizer.knowledge_db.combined_corpus_builder import (
    CombinedCorpusBuilder, field_has_corpus_layer)
from metaharmonizer.schema_cells import corpus_types, read_row_values, split_list


def test_split_list_keeps_empty_slots_on_request():
    assert split_list("NCIT:C1||NCIT:C3") == ("NCIT:C1", "NCIT:C3")
    assert split_list("NCIT:C1||NCIT:C3", keep_empty=True) == ("NCIT:C1", "", "NCIT:C3")
    assert split_list("NCIT:C1|NA|NCIT:C3", keep_empty=True) == ("NCIT:C1", "", "NCIT:C3")
    assert split_list("NA", keep_empty=True) == ()


def test_corpus_types_tokens():
    assert corpus_types("dynamic_enum|static_enum") == {"dynamic_enum", "static_enum"}
    assert corpus_types("dynamic_enum;static_enum") == {"dynamic_enum", "static_enum"}
    assert corpus_types("binary") == {"static_enum"}
    assert corpus_types("integer") == set() and corpus_types("NA") == set()


def test_custom_enum_list_is_read():
    rv = read_row_values("custom_enum", "PowerSoil DNA Isolation Kit|QIAamp", "NA")
    assert rv.labels == ("PowerSoil DNA Isolation Kit", "QIAamp") and rv.codes == ("", "")


def test_compound_row_keeps_its_ids():
    rv = read_row_values("dynamic_enum|static_enum", "Healthy|Control", "NCIT:C115935|NCIT:C1")
    assert rv.labels == ("Healthy", "Control") and rv.codes == ("NCIT:C115935", "NCIT:C1")


def test_empty_slot_keeps_the_pairing():
    rv = read_row_values("static_enum", "Stage I|Stage I/II|Stage II",
                         "NCIT:C27966||NCIT:C28054")
    assert rv.codes == ("NCIT:C27966", "", "NCIT:C28054")
    rv = read_row_values("static_enum", "A|B|C", "NCIT:C1|NA|NCIT:C3")
    assert rv.codes == ("NCIT:C1", "", "NCIT:C3")


def test_mismatched_counts_drop_ids_only():
    rv = read_row_values("static_enum", "A|B|C", "NCIT:C1|NCIT:C3")
    assert rv.labels == ("A", "B", "C") and rv.codes == ("", "", "")


def test_labels_split_on_pipe_only():
    rv = read_row_values("custom_enum", "AML with t(9;22)|CD4+ T cell|[unknown]", "NA")
    assert rv.labels == ("AML with t(9;22)", "CD4+ T cell", "[unknown]")


def test_non_list_types_have_no_labels():
    assert read_row_values("regexp", "[0-9]+|NA", "NA").labels == ()
    assert read_row_values("any", ".+", "NA").labels == ()
    assert read_row_values("dynamic_enum", "NA", "NA") == read_row_values("dynamic_enum", "", "")


def test_static_ids_without_labels_are_standalone_terms():
    rv = read_row_values("dynamic_enum|static_enum", "NA", "NCIT:C115935")
    assert rv.labels == () and rv.static_terms == ("NCIT:C115935",)


def test_custom_enum_ignores_static_enum():
    rv = read_row_values("custom_enum", "A|B", "NCIT:C1|NCIT:C2")
    assert rv.codes == ("", "")


def test_legacy_rows_unchanged():
    # No corpus.type: ;-separated for schema_vocab, |-separated for the builder.
    assert read_row_values("", "Female;Male", "NA").labels == ("Female", "Male")
    assert read_row_values("NA", "[0-9]+|NA", "NA").labels == ()
    assert read_row_values("", "vegan|omnivore", "NA", legacy_sep="|").labels == ("vegan", "omnivore")
    assert read_row_values("integer", "[0-9]+", "NA").labels == ()


REGISTRY = pd.DataFrame({
    "col.name": ["dna_extraction_kit", "tumor_stage", "disease", "treatment"],
    "multiplevalues": ["FALSE", "FALSE", "TRUE", "TRUE"],
    "corpus.type": ["custom_enum", "static_enum", "dynamic_enum|static_enum",
                    "dynamic_enum|static_enum"],
    "allowedvalues": ["PSP Spin Stool DNA Plus Kit|QIAamp", "Stage I|Stage I/II|Stage II",
                      "Healthy|Control", "NA"],
    "static.enum": ["NA", "NCIT:C27966||NCIT:C28054", "NCIT:C115935|NCIT:C1",
                    "NCIT:C41132"],
    "dynamic.enum": ["NA", "NA", "NCIT:C7057", "NCIT:C1908"],
    "dynamic.enum.property": ["NA", "NA", "descendant", "descendant"],
})


def test_schema_vocab_reads_registry_rows():
    v = sv.read_schema_table(REGISTRY)
    assert v["dna_extraction_kit"].allowed_values == ("PSP Spin Stool DNA Plus Kit", "QIAamp")
    assert v["tumor_stage"].allowed_codes == ("NCIT:C27966", "", "NCIT:C28054")
    assert v["disease"].allowed_codes == ("NCIT:C115935", "NCIT:C1")
    assert v["treatment"].static_terms == ("NCIT:C41132",) and v["treatment"].has_vocabulary


def test_standalone_static_term_joins_the_prefix_corpus(tmp_path):
    terms = {
        ("NCIT:C1908", "descendant"): [{"label": "Aspirin", "obo_id": "NCIT:C287"}],
        ("NCIT:C41132", "self"): [{"label": "None", "obo_id": "NCIT:C41132"}],
    }
    (c,) = sv.build_corpora(sv.read_schema_table(REGISTRY)["treatment"], tmp_path,
                            fetch=lambda r, rel: terms[(r, rel)])
    assert c.ontology_source == "ncit" and not c.local
    assert set(c.corpus_df["obo_id"]) == {"NCIT:C287", "NCIT:C41132"}


def test_builder_reads_rows_the_same_way():
    b = CombinedCorpusBuilder()
    rows = REGISTRY.set_index("col.name")
    recs = b.build_field_corpus(rows.loc["tumor_stage"].to_dict() | {"col.name": "tumor_stage"},
                                fetch_static_synonyms=False)
    assert [(r["label"], r["obo_id"]) for r in recs] == [
        ("Stage I", "NCIT:C27966"), ("Stage I/II", ""), ("Stage II", "NCIT:C28054")]
    recs = b.build_field_corpus({"col.name": "x", "corpus.type": "custom_enum",
                                 "allowedvalues": "CD4+ T cell|CD8+ T cell",
                                 "static.enum": "NA", "dynamic.enum": "NA"},
                                fetch_static_synonyms=False)
    assert [r["label"] for r in recs] == ["CD4+ T cell", "CD8+ T cell"]
    assert not field_has_corpus_layer({"corpus.type": "regexp", "allowedvalues": "A|B"})
