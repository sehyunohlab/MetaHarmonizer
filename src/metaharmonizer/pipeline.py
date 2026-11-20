"""End-to-end harmonization: SchemaMapper → OntologyMapper in one call.

:func:`harmonize` runs :class:`SchemaMapEngine` over every column of a
clinical table, picks the columns whose top-1 curated field is
ontology-backed (see :data:`FIELD_CATEGORY_PRESETS`), and runs
:class:`OntoMapEngine` over those columns' distinct values.

Columns that map to the same ``(corpus_category, ontology_source)`` are pooled
into a **single** OntoMapEngine run, so each corpus/model set is loaded once
rather than once per column.

The pipeline deliberately does **not** rewrite the input table. OntoMapEngine
always returns a best guess (it never abstains), so blindly substituting its
top-1 would silently replace values with confidently-wrong terms. Instead,
:attr:`HarmonizeResult.combined` gives one reviewable row per
``(source column, distinct value)`` with both the schema and ontology scores.

Example::

    from metaharmonizer import harmonize

    res = harmonize("clinical.tsv", schema="cbio")
    res.schema_map        # SchemaMapEngine output (one row per column)
    res.ontology_maps     # {(category, source): OntoMapEngine output}
    res.combined          # long table: column × value × ontology match
    res.harmonized        # long table: sample × curated field (merged)
    res.wide()            # one row per sample, <field> + <field>_ontology_term_id
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CategorySpec = Tuple[str, str]  # (corpus_category, ontology_source)

# Curated field → (corpus_category, ontology_source), per SchemaMapEngine
# preset. Only free-text, ontology-backed fields are listed; categorical
# fields with a closed value set (e.g. GDC ``treatment_type``, cBio
# ``sample_type``) are handled by SchemaMapper's value dict, not here.
# Every pair must be a key of OntoMapEngine's ``_CORPUS_REGISTRY``.
FIELD_CATEGORY_PRESETS: Dict[str, Dict[str, CategorySpec]] = {
    "cbio": {
        "disease": ("disease", "ncit"),
        "cancer_type": ("disease", "ncit"),
        "cancer_type_details": ("disease", "ncit"),
        "cancer_subtype": ("disease", "ncit"),
        "body_site": ("bodysite", "ncit"),
        "treatment_name": ("treatment", "ncit"),
    },
    "gdc": {
        "primary_diagnosis": ("disease", "ncit"),
        "disease_type": ("disease", "ncit"),
        "primary_site": ("bodysite", "ncit"),
        "tissue_or_organ_of_origin": ("bodysite", "ncit"),
        "site_of_resection_or_biopsy": ("bodysite", "ncit"),
        "therapeutic_agents": ("treatment", "ncit"),
    },
}

_EMPTY_STRINGS = {"", "nan", "none"}


@dataclass
class HarmonizeResult:
    """Outputs of :func:`harmonize`.

    Attributes:
        schema_map: SchemaMapEngine output, one row per input column.
        routing: One row per column sent to OntologyMapper, with its curated
            field, SchemaMapper score, and ``(corpus_category, ontology_source)``.
        ontology_maps: OntoMapEngine output keyed by ``(category, source)``;
            each holds the pooled distinct values of every column routed there.
        combined: Long table, one row per ``(source_column, value)``, joining
            ``routing`` to the matching OntoMapEngine row.
        skipped: Columns whose top-1 was ontology-backed but were not sent to
            OntologyMapper, with the ``reason`` (low schema score, numeric
            values, identifier-like values, or no values left after cleaning).
        errors: ``(category, source)`` → error message for any OntoMapEngine
            run that failed (the rest of the pipeline still completes).
        harmonized: Long table, one row per ``(input row, curated field)`` that
            received at least one ontology term. Several source columns routed
            to the same field are merged here (see :func:`harmonize`,
            ``multi_valued_fields``). Columns:

            - ``row``: positional index of the input row, then any ``id_columns``
            - ``curated_field``, ``corpus_category``, ``ontology_source``
            - ``value`` / ``term_id``: ``;``-joined labels and CURIEs, paired by
              position (``term_id`` is empty where the corpus had no code)
            - ``original_value`` / ``source_columns``: the kept source values
              and the columns they came from, paired by position
            - ``min_score``: lowest OntoMapEngine ``match1_score`` among kept
              values (OntoMapEngine never abstains, so treat low as suspect)
            - ``dropped``: ``col=value→label`` entries a single-valued field
              did not keep
            - ``conflict``: a dropped entry mapped to a term not kept
    """

    schema_map: pd.DataFrame
    routing: pd.DataFrame
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    ontology_maps: Dict[CategorySpec, pd.DataFrame] = field(default_factory=dict)
    combined: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: Dict[CategorySpec, str] = field(default_factory=dict)
    harmonized: pd.DataFrame = field(default_factory=pd.DataFrame)
    samples: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def wide(self, include_term_ids: bool = True) -> pd.DataFrame:
        """One row per input row, in the layout of a curated release.

        Columns: ``row``, any ``id_columns``, then per curated field ``<field>``
        (``;``-joined labels) and, if ``include_term_ids``, ``<field>_ontology_term_id``.
        Rows with no mapped term for a field hold NaN there.
        """
        out = self.samples.copy()
        if self.harmonized.empty:
            return out
        for f in self.harmonized["curated_field"].drop_duplicates():
            h = self.harmonized[self.harmonized["curated_field"] == f].set_index("row")
            out[f] = out["row"].map(h["value"])
            if include_term_ids:
                out[f"{f}_ontology_term_id"] = out["row"].map(h["term_id"])
        return out


# --- engine factories (module-level so tests can monkeypatch them) ---------

def _make_schema_engine(input_path: str, **kwargs):
    from metaharmonizer.models.schema_mapper import SchemaMapEngine
    return SchemaMapEngine(input_path, **kwargs)


def _make_onto_engine(category: str, query_ls: List[str], **kwargs):
    from metaharmonizer.engine.ontology_mapping_engine import OntoMapEngine
    return OntoMapEngine(category, query_ls, **kwargs)


def _noise_values() -> set:
    from metaharmonizer.models.schema_mapper.config import NOISE_VALUES
    return set(NOISE_VALUES)


# --- helpers ---------------------------------------------------------------

def _to_curie(code: str, ontology_source: str) -> str:
    """Invert OntoMapEngine's ``clean_code`` normalization.

    ``C1909`` (NCIt, prefix stripped) → ``NCIT:C1909``; ``UBERON_0001988`` →
    ``UBERON:0001988``; anything else is returned unchanged.
    """
    code = str(code).strip()
    if not code or code.lower() == "nan":
        return ""
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*)_(.+)$", code)
    if m:
        return f"{m.group(1).upper()}:{m.group(2)}"
    if ":" in code:
        return code
    if ontology_source.lower() == "ncit":
        return f"NCIT:{code}"
    return code


def _label_to_term_id(om, ontology_source: str) -> Dict[str, str]:
    """Lower-cased corpus label → CURIE, from the engine's normalized corpus.

    Labels shared by several codes keep the first code. Returns ``{}`` when
    the engine exposes no corpus (e.g. a test double).
    """
    params = getattr(om, "other_params", None)
    corpus = params.get("corpus_df") if isinstance(params, dict) else None
    if corpus is None or "official_label" not in getattr(corpus, "columns", ()):
        return {}
    if "clean_code" not in corpus.columns:
        return {}
    out: Dict[str, str] = {}
    for lab, code in zip(corpus["official_label"].astype(str), corpus["clean_code"]):
        out.setdefault(lab.strip().lower(), _to_curie(code, ontology_source))
    return out


def _build_harmonized(
    data: pd.DataFrame,
    routing: pd.DataFrame,
    combined: pd.DataFrame,
    multi_valued_fields: Iterable[str],
    noise: Iterable[str],
) -> pd.DataFrame:
    """Merge every routed column into one entry per (input row, curated field).

    Single-valued fields keep, per row, the highest-``schema_score`` column
    that has a mapped value for that row (ties: input column order); other
    columns with a value are recorded in ``dropped``. Multi-valued fields
    keep every column's value, de-duplicated by term.
    """
    cols_out = ["row", "curated_field", "corpus_category", "ontology_source",
                "value", "term_id", "original_value", "source_columns",
                "min_score", "dropped", "conflict"]
    if combined.empty:
        return pd.DataFrame(columns=cols_out)
    multi = set(multi_valued_fields)
    skip = {n.lower() for n in noise} | _EMPTY_STRINGS
    tid_col = "match1_term_id" if "match1_term_id" in combined.columns else None
    def _tid(r):
        t = r.get(tid_col) if tid_col else None
        return t if isinstance(t, str) else ""

    lookup = {
        (r["source_column"], r["value"]): (r.get("match1"), _tid(r), r.get("match1_score"))
        for _, r in combined.iterrows()
        if isinstance(r.get("match1"), str) and r.get("match1")
    }
    col_order = {c: i for i, c in enumerate(data.columns)}
    rows = []
    for fld, grp in routing.groupby("curated_field", sort=False):
        grp = grp.assign(_o=grp["source_column"].map(col_order)) \
                 .sort_values(["schema_score", "_o"], ascending=[False, True])
        cols = grp["source_column"].tolist()
        cat, src = grp["corpus_category"].iloc[0], grp["ontology_source"].iloc[0]
        series = {c: data[c].tolist() for c in cols}
        for i in range(len(data)):
            entries = []
            for c in cols:
                raw = series[c][i]
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    continue
                v = str(raw).strip()
                if v.lower() in skip:
                    continue
                hit = lookup.get((c, v))
                if hit is not None:
                    entries.append((c, v, *hit))
            if not entries:
                continue
            if fld in multi:
                kept, dropped = entries, []
            else:
                kept, dropped = entries[:1], entries[1:]
            seen, uniq = set(), []
            for e in kept:  # de-duplicate by term ID, else by label
                key = e[3] or e[2].lower()
                if key not in seen:
                    seen.add(key)
                    uniq.append(e)
            conflict = any((d[3] or d[2].lower()) not in seen for d in dropped)
            scores = [float(e[4]) for e in uniq if e[4] is not None and not pd.isna(e[4])]
            rows.append({
                "row": i, "curated_field": fld,
                "corpus_category": cat, "ontology_source": src,
                "value": ";".join(e[2] for e in uniq),
                "term_id": ";".join(e[3] for e in uniq),
                "original_value": ";".join(e[1] for e in uniq),
                "source_columns": ";".join(e[0] for e in uniq),
                "min_score": min(scores) if scores else None,
                "dropped": ";".join(f"{d[0]}={d[1]}→{d[2]}" for d in dropped),
                "conflict": conflict,
            })
    return pd.DataFrame(rows, columns=cols_out)


def _clean_values(series: pd.Series, noise: Iterable[str]) -> List[str]:
    """Distinct, stripped, non-empty, non-noise string values (order kept)."""
    noise = {n.lower() for n in noise}
    vals = series.dropna().astype(str).str.strip()
    keep = ~vals.str.lower().isin(_EMPTY_STRINGS | noise)
    return vals[keep].unique().tolist()


_ROUTE_COLS = ["source_column", "curated_field", "schema_score", "schema_stage",
               "corpus_category", "ontology_source"]


def _skip_reason(
    series: pd.Series,
    max_numeric_frac: float,
    max_unique_ratio: Optional[float],
    min_rows_for_unique_check: int = 20,
) -> Optional[str]:
    """Why a column's *values* are unfit for ontology mapping, else ``None``.

    Guards against SchemaMapper Stage-3 false positives that the score alone
    lets through: numeric columns (counts, scores) and identifier columns
    (patient/sample IDs, UUIDs) that are near-unique per row.
    """
    vals = series.dropna().astype(str).str.strip()
    vals = vals[~vals.str.lower().isin(_EMPTY_STRINGS)]
    if vals.empty:
        return None
    numeric_frac = pd.to_numeric(vals, errors="coerce").notna().mean()
    if numeric_frac > max_numeric_frac:
        return f"numeric values ({numeric_frac:.0%})"
    if max_unique_ratio is not None and len(vals) >= min_rows_for_unique_check:
        ratio = vals.nunique() / len(vals)
        if ratio > max_unique_ratio:
            return f"identifier-like values ({ratio:.0%} unique)"
    return None


def _route(
    schema_map: pd.DataFrame,
    field_categories: Mapping[str, CategorySpec],
    min_schema_score: float,
    data: Optional[pd.DataFrame] = None,
    max_numeric_frac: float = 0.5,
    max_unique_ratio: Optional[float] = 0.9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select columns whose SchemaMapper top-1 is an ontology-backed field.

    Returns ``(routing, skipped)``.
    """
    rows, skipped = [], []
    if schema_map.empty or "match1" not in schema_map.columns:
        return (pd.DataFrame(columns=_ROUTE_COLS),
                pd.DataFrame(columns=_ROUTE_COLS + ["reason"]))
    for _, r in schema_map.iterrows():
        target = r.get("match1")
        spec = field_categories.get(target) if isinstance(target, str) else None
        if spec is None:
            continue
        score = pd.to_numeric(r.get("match1_score"), errors="coerce")
        row = {
            "source_column": r["query"],
            "curated_field": target,
            "schema_score": None if pd.isna(score) else float(score),
            "schema_stage": r.get("stage"),
            "corpus_category": spec[0],
            "ontology_source": spec[1],
        }
        if pd.isna(score) or score < min_schema_score:
            reason = f"schema score {score} < {min_schema_score}"
        elif data is not None and r["query"] in data.columns:
            reason = _skip_reason(data[r["query"]], max_numeric_frac, max_unique_ratio)
        else:
            reason = None
        if reason:
            logger.info("[harmonize] skip %r -> %r: %s", r["query"], target, reason)
            skipped.append({**row, "reason": reason})
        else:
            rows.append(row)
    return (pd.DataFrame(rows, columns=_ROUTE_COLS),
            pd.DataFrame(skipped, columns=_ROUTE_COLS + ["reason"]))


# --- public API ------------------------------------------------------------

def harmonize(
    input_path: str,
    schema: Optional[str] = "cbio",
    *,
    field_categories: Optional[Mapping[str, CategorySpec]] = None,
    min_schema_score: float = 0.8,
    max_numeric_frac: float = 0.5,
    max_unique_ratio: Optional[float] = 0.9,
    drop_noise_values: bool = True,
    schema_kwargs: Optional[Dict[str, Any]] = None,
    onto_kwargs: Optional[Dict[str, Any]] = None,
    multi_valued_fields: Iterable[str] = (),
    id_columns: Optional[Sequence[str]] = None,
    raise_on_error: bool = False,
) -> HarmonizeResult:
    """Run SchemaMapper then OntologyMapper over one clinical table.

    Args:
        input_path: Clinical CSV/TSV (as for :class:`SchemaMapEngine`).
        schema: SchemaMapEngine preset (``"cbio"`` or ``"gdc"``). Also selects
            the default ``field_categories``. Defaults to ``"cbio"``.
        field_categories: Curated field → ``(corpus_category, ontology_source)``.
            Defaults to ``FIELD_CATEGORY_PRESETS[schema]``. Required when
            ``schema`` has no preset entry or a custom ``target_schema_path``
            is passed via ``schema_kwargs``.
        min_schema_score: Minimum SchemaMapper ``match1_score`` (0–1) for a
            column to be sent to OntologyMapper. Defaults to ``0.8``: Stage-1/2
            hits score ~1.0, while Stage-3 embedding matches in the 0.5–0.8
            band are mostly wrong (e.g. ``ICD_10`` → ``treatment_name``).
        max_numeric_frac: Skip a column when more than this fraction of its
            non-empty values parse as numbers. Defaults to ``0.5``.
        max_unique_ratio: Skip a column (with ≥20 non-empty values) when its
            distinct/non-empty ratio exceeds this, i.e. identifier-like
            (patient IDs, UUIDs). ``None`` disables. Defaults to ``0.9``.
        drop_noise_values: Drop SchemaMapper noise values (``unknown``,
            ``not reported``, ``na``…) before ontology mapping, so they are not
            forced onto an ontology term. Defaults to ``True``.
        schema_kwargs: Extra keyword args for :class:`SchemaMapEngine`
            (e.g. ``mode="auto"``, ``target_schema_path``, ``settings``).
        onto_kwargs: Extra keyword args for every :class:`OntoMapEngine` run
            (e.g. ``s3_strategy="rag"``, ``top_k``, ``output_dir``).
        multi_valued_fields: Curated fields that may hold several values per
            row (e.g. ``{"disease", "treatment"}``). When several source columns
            are routed to one of these, :attr:`HarmonizeResult.harmonized`
            keeps the union of their terms. Every other field is treated as
            single-valued: per row, the highest-scoring column that has a
            mapped value wins, and the rest are recorded in ``dropped``.
            Defaults to none (all single-valued).
        id_columns: Input columns to carry into :attr:`HarmonizeResult.harmonized`
            and :meth:`HarmonizeResult.wide` (e.g. ``["study_name", "sample_id"]``).
        raise_on_error: Re-raise an OntoMapEngine failure instead of recording
            it in :attr:`HarmonizeResult.errors`. Defaults to ``False``.

    Returns:
        HarmonizeResult
    """
    schema_kwargs = dict(schema_kwargs or {})
    onto_kwargs = dict(onto_kwargs or {})

    if field_categories is None:
        if "target_schema_path" in schema_kwargs:
            raise ValueError(
                "field_categories is required with a custom target_schema_path.")
        if schema not in FIELD_CATEGORY_PRESETS:
            raise ValueError(
                f"No field_categories preset for schema={schema!r}; pass "
                f"field_categories explicitly. Presets: {sorted(FIELD_CATEGORY_PRESETS)}.")
        field_categories = FIELD_CATEGORY_PRESETS[schema]

    # ---- Stage A: schema mapping -----------------------------------------
    sm = _make_schema_engine(input_path, schema=schema, **schema_kwargs)
    schema_map = sm.run_schema_mapping()
    data = sm.df.reset_index(drop=True)  # engine's already-loaded (dtype=str) table

    id_columns = list(id_columns or [])
    missing = [c for c in id_columns if c not in data.columns]
    if missing:
        raise ValueError(f"id_columns not in input: {missing}")
    samples = pd.concat([pd.DataFrame({"row": range(len(data))}),
                         data[id_columns].reset_index(drop=True)], axis=1)

    routing, skipped = _route(schema_map, field_categories, min_schema_score,
                              data, max_numeric_frac, max_unique_ratio)
    result = HarmonizeResult(schema_map=schema_map, routing=routing,
                             skipped=skipped, samples=samples)
    if routing.empty:
        logger.info("[harmonize] no columns routed to OntologyMapper")
        return result

    noise = _noise_values() if drop_noise_values else set()

    # ---- Stage B: ontology mapping, pooled per (category, source) ---------
    pieces = []
    for (category, source), grp in routing.groupby(
            ["corpus_category", "ontology_source"], sort=False):
        col_values = {c: _clean_values(data[c], noise) for c in grp["source_column"]}
        pooled = list(dict.fromkeys(v for vals in col_values.values() for v in vals))
        if not pooled:
            logger.info("[harmonize] (%s, %s): no values after cleaning", category, source)
            continue

        logger.info("[harmonize] OntoMapEngine(%s, %s): %d values from %d column(s)",
                    category, source, len(pooled), len(col_values))
        try:
            om = _make_onto_engine(category, pooled,
                                   ontology_source=source, **onto_kwargs)
            om_res = om.run()
        except Exception as exc:  # noqa: BLE001 — isolate one category's failure
            if raise_on_error:
                raise
            logger.exception("[harmonize] OntoMapEngine(%s, %s) failed", category, source)
            result.errors[(category, source)] = f"{type(exc).__name__}: {exc}"
            continue

        label_ids = _label_to_term_id(om, source)
        if label_ids and "match1" in om_res.columns:
            om_res = om_res.copy()
            om_res["match1_term_id"] = (om_res["match1"].astype(str).str.strip()
                                        .str.lower().map(label_ids).fillna(""))
        result.ontology_maps[(category, source)] = om_res
        om_by_value = om_res.rename(columns={"query": "value"})
        for _, route in grp.iterrows():
            vals = pd.DataFrame({"value": col_values[route["source_column"]]})
            for k in routing.columns:
                vals[k] = route[k]
            pieces.append(vals.merge(om_by_value, on="value", how="left"))

    if pieces:
        combined = pd.concat(pieces, ignore_index=True)
        lead = list(routing.columns) + ["value"]
        result.combined = combined[lead + [c for c in combined.columns if c not in lead]]

    # ---- Stage C: per-row merge across columns sharing a curated field ----
    h = _build_harmonized(data, routing, result.combined, multi_valued_fields, noise)
    if id_columns and not h.empty:
        h = h.merge(samples, on="row", how="left")
        h = h[["row"] + id_columns + [c for c in h.columns if c not in ["row"] + id_columns]]
    result.harmonized = h
    return result
