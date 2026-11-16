"""Prediction-invariance tests for Stage 2.5 (``_apply_synonym_boost``).

Stage 2.5 merges synonym-retrieval candidates into the Stage-2 candidate list
for low-confidence queries. Until 2026-11-11 it accepted that merge when
*either* the top-1 score improved *or* the gold rank improved::

    match_level = <rank of row['curated_ontology'] in combined_matches>
    boosted = (new_top1_score > old_top1_score
               or match_level < old_match_level)

So ``curated_ontology`` -- the answer -- selected predictions. Two distinct
failures followed from that one clause:

* **test** -- identical retrieval returned different candidates, and a
  different ``stage``, under different answer keys. The stage partition was
  therefore circular: a query landed in Stage 2.5 partly because Stage 2.5
  had got it right.
* **prod** -- ``ground_truth_map`` is the placeholder ``{query: "Not Found"}``,
  so the merged gold rank was always the 99 sentinel and ``99 < 99`` is False.
  The clause could never fire outside a benchmark, which means the benchmark
  was measuring a cascade that never shipped.

The contract these tests pin down: **candidates, scores and stage must not
depend on the answer key.** Scored correctness may change; predictions may
not. Companion to test_om_stage1_exact.py, which pins the same contract for
Stage 1.
"""
import logging

import pandas as pd
import pytest

from metaharmonizer.engine.ontology_mapping_engine import (
    SYNONYM_MIN_CONFIDENCE, OntoMapEngine)
from metaharmonizer.models.ontology_mapper_synonym import OntoMapSynonym

TOP_K = 3

# ── fixture: one low-confidence query per interesting shape ─────────────────
#
# Scores are strings, as the engine stores them (f"{x:.4f}").

STAGE2 = [
    # (query, [(label, score), ...])
    # A. merge does NOT raise top-1. The old gold clause fired here anyway
    #    whenever gold sat in the synonym list -- the regression case.
    ("senile cataract", [("juvenile cataract", "0.8200"),
                         ("cataract", "0.6100"),
                         ("cataract extraction", "0.4000")]),
    # B. merge DOES raise top-1: must still boost, gold or no gold.
    ("asthma attack", [("asthma-like symptom", "0.7000"),
                       ("wheeze", "0.6000"),
                       ("dyspnea", "0.5000")]),
    # C. high confidence: never enters the low-confidence mask at all.
    ("pain", [("Pain", "0.9500"),
              ("chronic pain", "0.7000"),
              ("ache", "0.6000")]),
    # D. no synonym row for this query: the boost must skip it.
    ("scoliosis of unknown cause", [("scoliosis", "0.8000"),
                                    ("kyphosis", "0.5000"),
                                    ("lordosis", "0.4000")]),
]

SYNONYMS = {
    "senile cataract": [("age-related cataract", "0.7000"),
                        ("cortical cataract", "0.5500"),
                        ("nuclear cataract", "0.3000")],
    "asthma attack": [("asthma", "0.9500"),
                      ("bronchial asthma", "0.8000"),
                      ("status asthmaticus", "0.4500")],
    "pain": [("pain sensation", "0.9900"),
             ("nociception", "0.8000"),
             ("soreness", "0.5000")],
}

# The true curations.
GOLD_ORIGINAL = {
    "senile cataract": "age-related cataract",   # synonym list, rank 2
    "asthma attack": "asthma",                   # synonym list, rank 1
    "pain": "Pain",                              # already stage-2 top-1
    "scoliosis of unknown cause": "scoliosis",
}

QUERIES = [q for q, _ in STAGE2]


def _frame(rows, gold):
    """Build a Stage-2 result frame exactly as the engine hands it over."""
    out = []
    for q, cands in rows:
        row = {"original_value": q,
               "curated_ontology": gold.get(q, "Not Found"),
               "match_level": 99,
               "stage": 2.0}
        for i, (lbl, scr) in enumerate(cands, start=1):
            row[f"match{i}"] = lbl
            row[f"match{i}_score"] = scr
        out.append(row)
    return pd.DataFrame(out)


def _syn_frame(gold, test_or_prod):
    out = []
    for q, cands in SYNONYMS.items():
        row = {"original_value": q,
               "curated_ontology": (gold.get(q, "Not Found")
                                    if test_or_prod == "test"
                                    else "Not Available for Prod Environment"),
               "match_level": 99}
        for i, (lbl, scr) in enumerate(cands, start=1):
            row[f"match{i}"] = lbl
            row[f"match{i}_score"] = scr
        out.append(row)
    return pd.DataFrame(out)


def _run(gold, test_or_prod="test"):
    """Run the real _apply_synonym_boost over the fixture; return the frame."""
    e = OntoMapEngine.__new__(OntoMapEngine)
    e._logger = logging.getLogger("test_om_stage25")
    e.top_k = TOP_K
    e.filter_obsolete = False
    e.ground_truth_map = gold
    e._test_or_prod = test_or_prod

    class _SynModel:
        def get_match_results(self, ground_truth_map, top_k, test_or_prod):
            return _syn_frame(ground_truth_map, test_or_prod)

    e._om_model_from_strategy = lambda strategy, queries: _SynModel()

    df = _frame(STAGE2, gold)
    e._apply_synonym_boost(df)
    return df


def _prediction(df):
    """The part of the output that must never depend on gold."""
    cols = (["original_value", "stage"]
            + [f"match{i}" for i in range(1, TOP_K + 1)]
            + [f"match{i}_score" for i in range(1, TOP_K + 1)])
    return (df[cols].astype(str)
                    .sort_values("original_value")
                    .to_records(index=False)
                    .tolist())


# ── the gold variants ───────────────────────────────────────────────────────

def _shuffled():
    """Each query gets another query's curation."""
    vals = [GOLD_ORIGINAL[q] for q in QUERIES]
    return dict(zip(QUERIES, vals[1:] + vals[:1]))


def _adversarial():
    """Gold placed exactly where the old clause would have fired: a synonym
    candidate that does NOT raise the top-1 score."""
    gold = dict(GOLD_ORIGINAL)
    gold["senile cataract"] = "cortical cataract"   # synonym rank 2, score 0.55
    return gold


GOLD_VARIANTS = {
    "original": (GOLD_ORIGINAL, "test"),
    "absent": ({q: "Not Found" for q in QUERIES}, "test"),
    "shuffled": (_shuffled(), "test"),
    "adversarial": (_adversarial(), "test"),
    "empty_map": ({}, "test"),
    "prod_placeholder": ({q: "Not Found" for q in QUERIES}, "prod"),
}


# ── 1. the contract ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", [v for v in GOLD_VARIANTS if v != "original"])
def test_predictions_invariant_to_gold(variant):
    """Candidates, scores and stage must be identical under any answer key."""
    gold, mode = GOLD_VARIANTS[variant]
    baseline = _prediction(_run(*GOLD_VARIANTS["original"]))
    assert _prediction(_run(gold, mode)) == baseline, (
        f"gold variant {variant!r} changed the predictions")


def test_all_variants_agree_pairwise():
    """One signature across every variant -- the invariance stated directly."""
    sigs = {name: _prediction(_run(gold, mode))
            for name, (gold, mode) in GOLD_VARIANTS.items()}
    distinct = {repr(v) for v in sigs.values()}
    assert len(distinct) == 1, f"{len(distinct)} distinct outputs: {sorted(sigs)}"


def test_stage_assignment_invariant_to_gold():
    """The stage partition must be well-defined, or per-stage accuracy is
    circular. Called out separately because this is what made the historical
    per-stage breakdown unrecoverable."""
    base = _run(*GOLD_VARIANTS["original"]).set_index("original_value")["stage"]
    for name, (gold, mode) in GOLD_VARIANTS.items():
        got = _run(gold, mode).set_index("original_value")["stage"]
        assert got.to_dict() == base.to_dict(), f"stage moved under {name!r}"


def test_gold_rank_does_not_rescue_a_non_improving_merge():
    """The regression itself. 'senile cataract' keeps its 0.8200 top-1 whether
    or not gold sits in the synonym list, so the merge must be rejected and the
    query must stay at stage 2.0."""
    for variant in ("original", "adversarial", "absent"):
        gold, mode = GOLD_VARIANTS[variant]
        row = _run(gold, mode).set_index("original_value").loc["senile cataract"]
        assert row["match1"] == "juvenile cataract"
        assert row["match2"] == "cataract", (
            f"{variant!r}: synonym candidate displaced a higher-scoring one")
        assert row["stage"] == 2.0


# ── 2. the stage still works (so the fix is not a silent disable) ───────────

def test_score_clause_still_boosts():
    """'asthma attack' has a synonym candidate at 0.9500 against a stage-2
    top-1 of 0.7000. That is a real signal and must still win."""
    row = _run(*GOLD_VARIANTS["original"]).set_index(
        "original_value").loc["asthma attack"]
    assert row["match1"] == "asthma"
    assert float(row["match1_score"]) == pytest.approx(0.95)
    assert row["stage"] == 2.5


def test_score_clause_boosts_without_any_gold():
    """The same boost must happen in prod, where there is no gold at all."""
    row = _run(*GOLD_VARIANTS["prod_placeholder"]).set_index(
        "original_value").loc["asthma attack"]
    assert row["match1"] == "asthma"
    assert row["stage"] == 2.5


def test_at_least_one_query_is_boosted_in_every_variant():
    """Guards against the fix degenerating into 'Stage 2.5 never fires'."""
    for name, (gold, mode) in GOLD_VARIANTS.items():
        df = _run(gold, mode)
        assert (df["stage"] == 2.5).any(), f"no boost at all under {name!r}"


# ── 3. the other shapes ─────────────────────────────────────────────────────

def test_high_confidence_query_is_untouched():
    """'pain' scores 0.9500, above the threshold, so it never enters the mask --
    even though its synonym list offers 0.9900."""
    assert SYNONYM_MIN_CONFIDENCE == 0.9
    row = _run(*GOLD_VARIANTS["original"]).set_index("original_value").loc["pain"]
    assert row["match1"] == "Pain"
    assert row["stage"] == 2.0


def test_query_with_no_synonym_row_is_untouched():
    row = _run(*GOLD_VARIANTS["original"]).set_index(
        "original_value").loc["scoliosis of unknown cause"]
    assert row["match1"] == "scoliosis"
    assert row["stage"] == 2.0


def test_obsolete_filter_still_applies_and_is_gold_free():
    """filter_obsolete drops obsolete_ synonym candidates; that decision must
    also not consult gold."""
    def run(gold, mode):
        e = OntoMapEngine.__new__(OntoMapEngine)
        e._logger = logging.getLogger("test_om_stage25")
        e.top_k = TOP_K
        e.filter_obsolete = True
        e.ground_truth_map = gold
        e._test_or_prod = mode

        syn = pd.DataFrame([{
            "original_value": "asthma attack", "match_level": 99,
            "curated_ontology": gold.get("asthma attack", "Not Found"),
            "match1": "obsolete_asthma", "match1_score": "0.9900",
            "match2": "asthma", "match2_score": "0.9500",
            "match3": None, "match3_score": "0.0000"}])

        class _M:
            def get_match_results(self, **kwargs):
                return syn.copy()

        e._om_model_from_strategy = lambda *a: _M()
        df = _frame([STAGE2[1]], gold)
        e._apply_synonym_boost(df)
        return df.iloc[0]

    for gold, mode in (({"asthma attack": "obsolete_asthma"}, "test"),
                       ({"asthma attack": "asthma"}, "test"),
                       ({"asthma attack": "Not Found"}, "prod")):
        row = run(gold, mode)
        assert row["match1"] == "asthma", "obsolete candidate was not filtered"


# ── 4. the audited second channel ───────────────────────────────────────────

def test_synonym_retriever_candidates_are_gold_free():
    """``_apply_synonym_boost`` passes ``ground_truth_map`` into the synonym
    retriever (engine line ~1214). Audited 2026-11-10 as scoring-only:
    ``search_many`` runs before gold is read, and gold only fills the
    ``curated_ontology``/``match_level`` columns, which the boost discards.
    Pinned here so a future edit cannot quietly make it a live channel."""
    m = OntoMapSynonym.__new__(OntoMapSynonym)
    m.query = ["senile cataract"]
    m.top_k = 2
    m.logger = logging.getLogger("test_om_stage25")

    class _Dict:
        def search_many(self, queries, top_k):
            return [[{"official_label": "age-related cataract", "score": 0.70},
                     {"official_label": "cortical cataract", "score": 0.55}]
                    for _ in queries]

    m.syn_dict = _Dict()

    cand_cols = ["match1", "match1_score", "match2", "match2_score"]
    sigs = set()
    for gold, mode in (({"senile cataract": "age-related cataract"}, "test"),
                       ({"senile cataract": "cortical cataract"}, "test"),
                       ({"senile cataract": "Not Found"}, "test"),
                       ({}, "prod")):
        df = m.get_match_results(ground_truth_map=gold, test_or_prod=mode)
        sigs.add(tuple(df[cand_cols].astype(str).iloc[0]))
    assert len(sigs) == 1, "synonym retrieval varied with the answer key"
