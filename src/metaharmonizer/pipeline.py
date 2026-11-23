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
    res.wide()            # one row per sample: <field> + <field>_ontology_term_id,
                          # unmapped fields passed through; provenance=True adds
                          # <field>_source_column / _schema_score / _map_score …

With a schema table (``schema_table=``, e.g. a data dictionary with
``DynamicEnum`` / ``AllowedValues`` columns) the table is the single source:
it is the SchemaMapper target schema, and each field's vocabulary becomes its
OntologyMapper corpus (see :mod:`metaharmonizer.schema_vocab`), so no
``field_categories`` is needed::

    res = harmonize("sampleMetadata.csv", schema=None,
                    schema_table="cMD_data_dictionary.csv")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

CategorySpec = Tuple[str, str]  # (corpus_category, ontology_source)
# A field may be served by several corpora (e.g. roots in two ontologies, or
# an ontology plus a local list); each value then keeps its best-scoring match.
FieldSpec = Union[CategorySpec, Sequence[CategorySpec]]

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
        passthrough_routing: Columns whose SchemaMapper top-1 is a field with
            no vocabulary (not in ``field_categories``), scoring at least
            ``min_schema_score``. :meth:`wide` copies their values as-is.
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
    passthrough_routing: pd.DataFrame = field(default_factory=pd.DataFrame)
    samples: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    source_data: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    field_order: List[str] = field(default_factory=list, repr=False)
    multi_valued_fields: List[str] = field(default_factory=list, repr=False)

    def wide(self, include_term_ids: bool = True, passthrough: bool = True,
             provenance: bool = False) -> pd.DataFrame:
        """One row per input row, in the layout of a curated release.

        Columns: ``row``, any ``id_columns``, then per curated field, in schema
        order (``schema_table`` field order when given):

        - ``<field>``: ``;``-joined labels for ontology-mapped fields; for a
          field without a vocabulary (``passthrough``), the source value as-is
          from the column SchemaMapper mapped there (see
          :attr:`passthrough_routing`).
        - ``<field>_ontology_term_id`` (``include_term_ids``, mapped fields
          that have any CURIE; ``AllowedValues`` labels have none).

        With ``provenance=True``, also:

        - ``<field>_source_column``: source column(s) the value came from
        - ``<field>_schema_score``: SchemaMapper score of that column (lowest
          when several)
        - mapped fields only: ``<field>_original_value`` (the source values,
          paired by position with the labels), ``<field>_map_score``
          (lowest OntoMapEngine ``match1_score``) and, if any row has one,
          ``<field>_dropped`` (values a single-valued field did not keep).

        Rows with nothing for a field hold NaN there.
        """
        out = self.samples.copy()
        rows = out["row"]
        blocks: Dict[str, Dict[str, pd.Series]] = {}

        h = self.harmonized
        if not h.empty:
            col_score = {}
            if not self.routing.empty:
                for r in self.routing.drop_duplicates(["curated_field", "source_column"]).itertuples():
                    col_score[(r.curated_field, r.source_column)] = r.schema_score
            for f in h["curated_field"].drop_duplicates():
                g = h[h["curated_field"] == f].set_index("row")
                b = {f: rows.map(g["value"])}
                has_ids = g["term_id"].astype(str).str.replace(";", "", regex=False).str.strip().ne("").any()
                if include_term_ids and has_ids:
                    b[f"{f}_ontology_term_id"] = rows.map(g["term_id"])
                if provenance:
                    score = g["source_columns"].map(lambda s, f=f: min(
                        (col_score.get((f, c), float("nan")) for c in str(s).split(";")),
                        default=float("nan")))
                    b[f"{f}_source_column"] = rows.map(g["source_columns"])
                    b[f"{f}_original_value"] = rows.map(g["original_value"])
                    b[f"{f}_schema_score"] = rows.map(score)
                    b[f"{f}_map_score"] = rows.map(g["min_score"])
                    if g["dropped"].ne("").any():
                        b[f"{f}_dropped"] = rows.map(g["dropped"].where(g["dropped"].ne("")))
                blocks[f] = b

        pr = self.passthrough_routing
        if passthrough and not pr.empty and not self.source_data.empty:
            pr = pr.sort_values("schema_score", ascending=False, kind="stable")
            multi = set(self.multi_valued_fields)
            for f, grp in pr.groupby("curated_field", sort=False):
                if f in blocks or f in out.columns:
                    continue
                cols = [c for c in grp["source_column"] if c in self.source_data.columns]
                if not cols:
                    continue
                score = dict(zip(grp["source_column"], grp["schema_score"]))
                cand = pd.DataFrame({c: _clean_cells(self.source_data[c]) for c in cols})
                present = cand.notna()
                if f in multi:
                    value = cand.apply(lambda r: ";".join(r.dropna()), axis=1)
                    src = present.apply(lambda r: ";".join(r.index[r]), axis=1)
                    value, src = value.where(value != ""), src.where(src != "")
                else:
                    value = cand.bfill(axis=1).iloc[:, 0]
                    src = present.idxmax(axis=1).where(present.any(axis=1))
                b = {f: value.values}
                if provenance:
                    b[f"{f}_source_column"] = src.values
                    b[f"{f}_schema_score"] = src.map(lambda s: min(
                        (score[c] for c in str(s).split(";")), default=float("nan"))
                        if isinstance(s, str) else float("nan")).values
                blocks[f] = b

        order = [f for f in self.field_order if f in blocks]
        order += [f for f in blocks if f not in order]
        new = {name: col for f in order for name, col in blocks[f].items()}
        if not new:
            return out
        return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


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
        (r["source_column"], r["value"]): (r.get("match1"), _tid(r), r.get("match1_score"),
                                           r.get("corpus_category"), r.get("ontology_source"))
        for _, r in combined.iterrows()
        if isinstance(r.get("match1"), str) and r.get("match1")
    }
    col_order = {c: i for i, c in enumerate(data.columns)}
    rows = []
    routing = routing.drop_duplicates(["curated_field", "source_column"])
    for fld, grp in routing.groupby("curated_field", sort=False):
        grp = grp.assign(_o=grp["source_column"].map(col_order)) \
                 .sort_values(["schema_score", "_o"], ascending=[False, True])
        cols = grp["source_column"].tolist()
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
                "corpus_category": ";".join(dict.fromkeys(str(e[5]) for e in uniq)),
                "ontology_source": ";".join(dict.fromkeys(str(e[6]) for e in uniq)),
                "value": ";".join(e[2] for e in uniq),
                "term_id": ";".join(e[3] for e in uniq),
                "original_value": ";".join(e[1] for e in uniq),
                "source_columns": ";".join(e[0] for e in uniq),
                "min_score": min(scores) if scores else None,
                "dropped": ";".join(f"{d[0]}={d[1]}→{d[2]}" for d in dropped),
                "conflict": conflict,
            })
    return pd.DataFrame(rows, columns=cols_out)


def _best_per_value(combined: pd.DataFrame) -> pd.DataFrame:
    """Keep, per ``(source_column, value)``, the row with the highest
    ``match1_score`` (a column routed to several corpora gets one answer)."""
    if combined.empty or not combined.duplicated(["source_column", "value"]).any():
        return combined
    score = pd.to_numeric(combined.get("match1_score"), errors="coerce").fillna(-1.0)
    order = combined.assign(_s=score, _i=range(len(combined))) \
                    .sort_values(["_s", "_i"], ascending=[False, True])
    best = order.drop_duplicates(["source_column", "value"]).sort_values("_i")
    return best.drop(columns=["_s", "_i"]).reset_index(drop=True)


def _clean_values(series: pd.Series, noise: Iterable[str]) -> List[str]:
    """Distinct, stripped, non-empty, non-noise string values (order kept)."""
    noise = {n.lower() for n in noise}
    vals = series.dropna().astype(str).str.strip()
    keep = ~vals.str.lower().isin(_EMPTY_STRINGS | noise)
    return vals[keep].unique().tolist()


_ROUTE_COLS = ["source_column", "curated_field", "schema_score", "schema_stage",
               "corpus_category", "ontology_source"]
_PASS_COLS = ["source_column", "curated_field", "schema_score", "schema_stage"]


def _clean_cells(series: pd.Series) -> pd.Series:
    """Stripped string values, with empty / ``nan`` / ``none`` cells as NaN."""
    s = series.astype("string").str.strip()
    return s.where(s.notna() & ~s.str.lower().isin(_EMPTY_STRINGS)).astype(object)


def _passthrough_route(schema_map: pd.DataFrame, field_categories: Mapping[str, Any],
                       min_schema_score: float, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Columns whose SchemaMapper top-1 is a field *without* a vocabulary,
    scoring at least ``min_schema_score``: their values are carried over
    as-is by :meth:`HarmonizeResult.wide`."""
    if schema_map.empty or "match1" not in schema_map.columns:
        return pd.DataFrame(columns=_PASS_COLS)
    exclude = set(exclude)
    score = pd.to_numeric(schema_map["match1_score"], errors="coerce")
    keep = (schema_map["match1"].map(lambda t: isinstance(t, str) and t not in field_categories
                                     and t not in exclude)
            & (score >= min_schema_score))
    sm = schema_map[keep]
    return pd.DataFrame({"source_column": sm["query"].values, "curated_field": sm["match1"].values,
                         "schema_score": score[keep].astype(float).values,
                         "schema_stage": sm["stage"].values if "stage" in sm else None},
                        columns=_PASS_COLS)


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


def _specs(spec: FieldSpec) -> List[CategorySpec]:
    """A field's corpora as a list of ``(category, source)`` pairs."""
    if (isinstance(spec, tuple) and len(spec) == 2
            and all(isinstance(x, str) for x in spec)):
        return [spec]
    return [tuple(s) for s in spec]


def _route(
    schema_map: pd.DataFrame,
    field_categories: Mapping[str, FieldSpec],
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
        base = {
            "source_column": r["query"],
            "curated_field": target,
            "schema_score": None if pd.isna(score) else float(score),
            "schema_stage": r.get("stage"),
        }
        if pd.isna(score) or score < min_schema_score:
            reason = f"schema score {score} < {min_schema_score}"
        elif data is not None and r["query"] in data.columns:
            reason = _skip_reason(data[r["query"]], max_numeric_frac, max_unique_ratio)
        else:
            reason = None
        if reason:
            logger.info("[harmonize] skip %r -> %r: %s", r["query"], target, reason)
        for cat, src in _specs(spec):
            row = {**base, "corpus_category": cat, "ontology_source": src}
            if reason:
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
    field_categories: Optional[Mapping[str, FieldSpec]] = None,
    schema_table: Optional[Union[str, Path, pd.DataFrame]] = None,
    vocab_cache_dir: Optional[Union[str, Path]] = None,
    min_schema_score: float = 0.8,
    max_numeric_frac: float = 0.5,
    max_unique_ratio: Optional[float] = 0.9,
    drop_noise_values: bool = True,
    schema_kwargs: Optional[Dict[str, Any]] = None,
    onto_kwargs: Optional[Dict[str, Any]] = None,
    multi_valued_fields: Optional[Iterable[str]] = None,
    id_columns: Optional[Sequence[str]] = None,
    raise_on_error: bool = False,
) -> HarmonizeResult:
    """Run SchemaMapper then OntologyMapper over one clinical table.

    Args:
        input_path: Clinical CSV/TSV (as for :class:`SchemaMapEngine`).
        schema: SchemaMapEngine preset (``"cbio"`` or ``"gdc"``). Also selects
            the default ``field_categories``. Defaults to ``"cbio"``.
        field_categories: Curated field → ``(corpus_category, ontology_source)``,
            or a list of such pairs to search several corpora (each value
            keeps its best-scoring match). Defaults to
            ``FIELD_CATEGORY_PRESETS[schema]``, or to the corpora derived from
            ``schema_table``. Required when neither applies (e.g. a custom
            ``target_schema_path`` in ``schema_kwargs``).
        schema_table: A schema table (path or DataFrame), e.g. a data
            dictionary with ``ColName``, ``ColClass``, ``MultipleValues``,
            ``DynamicEnum``, ``DynamicEnumProperty`` and ``AllowedValues``
            columns. Used as the SchemaMapper target schema (unless
            ``schema_kwargs`` sets ``target_schema_path``; NCIt roots become
            its ``ontology_root``), and each field's ``DynamicEnum`` roots and
            ``AllowedValues`` labels become its OntologyMapper corpora (see
            :mod:`metaharmonizer.schema_vocab`). Fields with neither are not
            sent to OntologyMapper. Allowed-value labels outside any ontology
            get an empty ``term_id``.
        vocab_cache_dir: Where corpora built from ``schema_table`` roots are
            cached. Defaults to ``<data dir>/corpus/schema_vocab``.
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
            Defaults to the ``MultipleValues`` column of ``schema_table``,
            else none (all single-valued).
        id_columns: Input columns to carry into :attr:`HarmonizeResult.harmonized`
            and :meth:`HarmonizeResult.wide` (e.g. ``["study_name", "sample_id"]``).
        raise_on_error: Re-raise an OntoMapEngine failure instead of recording
            it in :attr:`HarmonizeResult.errors`. Defaults to ``False``.

    Returns:
        HarmonizeResult
    """
    schema_kwargs = dict(schema_kwargs or {})
    onto_kwargs = dict(onto_kwargs or {})
    corpora: Dict[str, Any] = {}   # corpus_category -> VocabCorpus
    field_order: List[str] = []

    if schema_table is not None:
        from metaharmonizer import schema_vocab as sv
        from metaharmonizer._paths import CORPUS_DIR
        cache = Path(vocab_cache_dir) if vocab_cache_dir else CORPUS_DIR / "schema_vocab"
        table = sv.load_schema_table(schema_table)
        vocabs = sv.read_schema_table(table)
        field_order = list(vocabs)
        if "target_schema_path" not in schema_kwargs:
            ts = sv.target_schema_frame(table)
            cache.mkdir(parents=True, exist_ok=True)
            import hashlib
            digest = hashlib.sha1(ts.to_csv(index=False).encode()).hexdigest()[:10]
            ts_path = cache / f"target_schema_{digest}.csv"
            ts.to_csv(ts_path, index=False)
            schema_kwargs["target_schema_path"] = str(ts_path)
        if field_categories is None:
            fc = sv.build_field_corpora(vocabs, cache)
            field_categories = {f: [(c.key, c.ontology_source) for c in cs]
                                for f, cs in fc.items()}
            corpora = {c.key: c for cs in fc.values() for c in cs}
            logger.info("[harmonize] %d fields with a vocabulary, %d corpora",
                        len(field_categories), len(corpora))
        if multi_valued_fields is None:
            multi_valued_fields = [f for f, v in vocabs.items() if v.multiple_values]
    multi_valued_fields = list(multi_valued_fields or ())

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
    passthrough = _passthrough_route(schema_map, field_categories, min_schema_score,
                                     exclude=id_columns)
    result = HarmonizeResult(schema_map=schema_map, routing=routing,
                             skipped=skipped, samples=samples,
                             passthrough_routing=passthrough, source_data=data,
                             field_order=field_order,
                             multi_valued_fields=multi_valued_fields)
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
        vc = corpora.get(category)
        kw = dict(onto_kwargs)
        if vc is not None:
            kw["corpus_df"] = vc.corpus_df
        try:
            om = _make_onto_engine(category, pooled, ontology_source=source, **kw)
            om_res = om.run()
        except Exception as exc:  # noqa: BLE001 — isolate one category's failure
            if raise_on_error:
                raise
            logger.exception("[harmonize] OntoMapEngine(%s, %s) failed", category, source)
            result.errors[(category, source)] = f"{type(exc).__name__}: {exc}"
            continue

        label_ids = {} if (vc is not None and vc.local) else _label_to_term_id(om, source)
        if "match1" in om_res.columns and (label_ids or (vc is not None and vc.local)):
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
        combined = _best_per_value(pd.concat(pieces, ignore_index=True))
        lead = list(routing.columns) + ["value"]
        result.combined = combined[lead + [c for c in combined.columns if c not in lead]]

    # ---- Stage C: per-row merge across columns sharing a curated field ----
    h = _build_harmonized(data, routing, result.combined, multi_valued_fields, noise)
    if id_columns and not h.empty:
        h = h.merge(samples, on="row", how="left")
        h = h[["row"] + id_columns + [c for c in h.columns if c not in ["row"] + id_columns]]
    result.harmonized = h
    return result
