"""Per-field value vocabularies read from a schema table.

A schema table (a *data dictionary*) lists the curated fields and, for each,
the vocabulary its values come from. Two columns carry that vocabulary:

- ``DynamicEnum`` (alias ``ontology_root``): one or more ontology root CURIEs,
  ``;`` or ``|`` separated, e.g. ``NCIT:C7057;EFO:0000408``. With
  ``DynamicEnumProperty`` = ``descendant`` (default) the vocabulary is every
  descendant of each root plus the root; with ``children`` it is the direct
  children only.
- ``AllowedValues``: permitted labels. With a ``corpus.type`` column (the
  schema registry convention), a ``static_enum`` or ``custom_enum`` row holds a
  ``|`` list (``Female|Male``) and ``static.enum`` the CURIE of each label, in
  the same order, with an empty slot for a label without one; other rows hold
  no labels. A ``static_enum`` row with IDs but no labels adds those terms as
  they are (``Healthy``: ``NCIT:C115935``). Without ``corpus.type`` (the older
  cMD layout) the list is ``;``-separated, e.g. ``Female;Male``, and a value
  that is a regular expression (``[0-9]+``, ``.+``) describes a format, not a
  vocabulary, and is ignored. See :mod:`metaharmonizer.schema_cells`, which
  :class:`~metaharmonizer.knowledge_db.combined_corpus_builder.CombinedCorpusBuilder`
  shares.

:func:`read_schema_table` parses such a table into :class:`FieldVocab` records,
:func:`target_schema_frame` turns it into the target-schema CSV layout that
:class:`SchemaMapEngine` reads, and :func:`build_corpora` turns a
:class:`FieldVocab` into OntoMapEngine corpora (one per ontology prefix, plus
one local corpus for allowed values the roots do not already contain).

OntoMapEngine accepts one ontology prefix per corpus, so a field with roots in
two ontologies gets two corpora. Terms an ontology imports from another one
(OLS returns those as descendants too, e.g. AfPO terms under HANCESTRO) are
dropped: each root keeps only terms carrying its own prefix.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from metaharmonizer.schema_cells import REGEX_SIGNS, _legacy_labels, cell as _cell, read_row_values

logger = logging.getLogger(__name__)

# Column aliases: data-dictionary names (CamelCase as in the cMD dictionary,
# dotted lower case as in the cBioPortal one), then MetaHarmonizer's names.
_NAME_COLS = ("ColName", "col.name", "field_name")
_ROOT_COLS = ("DynamicEnum", "dynamic.enum", "ontology_root")
_PROP_COLS = ("DynamicEnumProperty", "dynamic.enum.property", "ontology_root_property")
_ALLOWED_COLS = ("AllowedValues", "allowedvalues", "allowed_values")
_MULTI_COLS = ("MultipleValues", "multiplevalues", "multiple_values")
_CLASS_COLS = ("ColClass", "col.class")
# Optional: ``corpus.type`` decides how AllowedValues is read (see
# metaharmonizer.schema_cells); ``static.enum`` holds the term CURIEs of each
# label, in the same order (e.g. age_group in the cBioPortal dictionary).
_CTYPE_COLS = ("CorpusType", "corpus.type")
_STATIC_COLS = ("StaticEnum", "static.enum")
_NUMERIC_CLASSES = {"integer", "numeric", "double", "float", "number"}

# Prefix of the codes given to allowed-value labels that have no ontology
# code. OntoMapEngine treats a corpus whose code prefix it cannot resolve as a
# local corpus (no concept tables); harmonize() leaves their term_id empty.
LOCAL_CODE_PREFIX = "LOCAL"



@dataclass(frozen=True)
class FieldVocab:
    """Vocabulary of one curated field."""

    field: str
    roots: Tuple[str, ...] = ()
    root_property: str = "descendant"
    allowed_values: Tuple[str, ...] = ()
    multiple_values: bool = False
    # CURIEs of allowed_values, paired by position ("" where none was given)
    allowed_codes: Tuple[str, ...] = ()
    # static.enum CURIEs given without labels: each term is added as it is
    static_terms: Tuple[str, ...] = ()

    @property
    def has_vocabulary(self) -> bool:
        return bool(self.roots or self.allowed_values or self.static_terms)


@dataclass
class VocabCorpus:
    """One OntoMapEngine corpus serving one or more fields."""

    key: str                    # safe identifier, used as corpus_category
    ontology_source: str        # e.g. "ncit", "hancestro", "local"
    corpus_df: pd.DataFrame = field(repr=False)
    local: bool = False         # codes are LOCAL_*, not real ontology codes
    # local corpus only: lower-cased label -> CURIE the schema gave it
    # (``static.enum``); labels without one have no term ID
    term_ids: Dict[str, str] = field(default_factory=dict, repr=False)


def _pick(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    return next((c for c in names if c in df.columns), None)


def parse_allowed_values(cell) -> Tuple[str, ...]:
    """Labels from a ``;``-separated ``AllowedValues`` cell (the layout without
    ``corpus.type``), or ``()`` for a pattern.

    The whole cell is treated as a pattern (and ignored) when any item looks
    like a regular expression, e.g. ``[0-9]+|NA`` or ``.+``.
    """
    return tuple(dict.fromkeys(_legacy_labels(cell, ";")))


def _parse_roots(cell) -> Tuple[str, ...]:
    s = _cell(cell)
    if not s:
        return ()
    return tuple(dict.fromkeys(t.strip() for t in re.split(r"[;|]", s) if t.strip()))


def _to_bool(v) -> bool:
    return _cell(v).lower() in {"true", "t", "yes", "y", "1"}


def load_schema_table(schema_table: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(schema_table, pd.DataFrame):
        return schema_table.copy()
    return pd.read_csv(schema_table, dtype=str, keep_default_na=False)


def read_schema_table(schema_table: Union[str, Path, pd.DataFrame]) -> Dict[str, FieldVocab]:
    """Field name → :class:`FieldVocab` for every row of a schema table."""
    df = load_schema_table(schema_table)
    name = _pick(df, _NAME_COLS)
    if name is None:
        raise ValueError(f"schema table needs a field-name column {_NAME_COLS}; "
                         f"has {list(df.columns)}")
    root_c, prop_c = _pick(df, _ROOT_COLS), _pick(df, _PROP_COLS)
    allow_c, multi_c = _pick(df, _ALLOWED_COLS), _pick(df, _MULTI_COLS)
    ctype_c, static_c = _pick(df, _CTYPE_COLS), _pick(df, _STATIC_COLS)
    out: Dict[str, FieldVocab] = {}
    for _, r in df.iterrows():
        f = _cell(r[name])
        if not f:
            continue
        prop = (_cell(r[prop_c]) if prop_c else "").lower() or "descendant"
        if prop not in {"descendant", "children"}:
            logger.warning("[schema_vocab] %s: unknown root property %r; using descendant", f, prop)
            prop = "descendant"
        rv = read_row_values(r[ctype_c] if ctype_c else "",
                             r[allow_c] if allow_c else "",
                             r[static_c] if static_c else "", f)
        out[f] = FieldVocab(
            field=f,
            roots=_parse_roots(r[root_c]) if root_c else (),
            root_property=prop,
            allowed_values=rv.labels,
            multiple_values=_to_bool(r[multi_c]) if multi_c else False,
            allowed_codes=rv.codes,
            static_terms=rv.static_terms,
        )
    return out


def target_schema_frame(schema_table: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """The table in SchemaMapEngine's target-schema layout.

    Columns ``field_name``, ``is_numeric_field`` (from ``ColClass`` when
    present) and ``ontology_root`` (the NCIt roots only, ``|``-joined, since
    SchemaMapper's Stage 2 searches NCIt).
    """
    df = load_schema_table(schema_table)
    if "field_name" in df.columns and "ColName" not in df.columns:
        return df  # already in SchemaMapEngine layout
    vocab = read_schema_table(df)
    cls_c = _pick(df, _CLASS_COLS)
    name = _pick(df, _NAME_COLS)
    cls = dict(zip(df[name].map(_cell), df[cls_c].map(_cell))) if cls_c else {}
    rows = []
    for f, v in vocab.items():
        ncit = [r for r in v.roots if r.upper().startswith("NCIT:")]
        rows.append({
            "field_name": f,
            "is_numeric_field": "yes" if cls.get(f, "").lower() in _NUMERIC_CLASSES else "no",
            "ontology_root": "|".join(ncit),
        })
    return pd.DataFrame(rows)


# --- corpora -----------------------------------------------------------------

def _safe(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


def _fetch_terms(root: str, relation: str) -> List[dict]:
    """Descendants or children of a root from OLS (the root itself excluded:
    it names the category, e.g. ``Menopause Status``, not a value), or with
    ``relation="self"`` the term itself (a standalone ``static.enum`` term). An
    obsolete root yields nothing."""
    from metaharmonizer.knowledge_db.corpus_builder import CorpusBuilder
    from metaharmonizer.knowledge_db.db_clients.ols_db import OLSDb
    from metaharmonizer._async_utils import run_async

    ols, parse = OLSDb(), CorpusBuilder._parse_term

    async def go():
        root_raw = await ols.get_term(root)
        if root_raw.get("is_obsolete"):
            logger.warning("[schema_vocab] root %s is obsolete in OLS (%r); skipped",
                           root, root_raw.get("label"))
            return []
        if relation == "self":
            return [parse(root_raw)]
        get = ols.get_children if relation == "children" else ols.get_descendants
        rel = await get(root)
        return [parse(t) for t in rel]

    return run_async(go())


def root_corpus(root: str, relation: str = "descendant",
                cache_dir: Optional[Path] = None,
                fetch: Optional[Callable[[str, str], List[dict]]] = None) -> pd.DataFrame:
    """Terms under one root, restricted to the root's own prefix; cached as CSV."""
    prefix = root.split(":", 1)[0].upper()
    path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{_safe(root)}__{relation}.csv"
        if path.exists():
            return pd.read_csv(path, dtype=str, keep_default_na=False)
    recs = (fetch or _fetch_terms)(root, relation)
    df = pd.DataFrame(recs, columns=["label", "obo_id", "synonyms", "description"]) \
        if recs else pd.DataFrame(columns=["label", "obo_id", "synonyms", "description"])
    df = df[df["obo_id"].astype(str).str.upper().str.startswith(prefix + ":")]
    df = df[df["label"].astype(str).str.strip() != ""]
    df = df.drop_duplicates("obo_id").reset_index(drop=True)
    if path is not None:
        df.to_csv(path, index=False)
    logger.info("[schema_vocab] %s (%s): %d %s terms", root, relation, len(df), prefix)
    return df


def build_corpora(vocab: FieldVocab, cache_dir: Optional[Path] = None,
                  fetch: Optional[Callable[[str, str], List[dict]]] = None) -> List[VocabCorpus]:
    """OntoMapEngine corpora for one field.

    One corpus per ontology prefix among the field's roots and standalone
    ``static.enum`` terms, keyed by them (so fields with the same roots share
    it), plus a local corpus for allowed values whose label is not already in
    those corpora.
    """
    by_prefix: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    sources = [(r, vocab.root_property, r) for r in vocab.roots]
    sources += [(t, "self", "s_" + t) for t in vocab.static_terms if t not in vocab.roots]
    for root, relation, name in sources:
        if ":" not in root:
            logger.warning("[schema_vocab] %s: %r is not a CURIE; skipped", vocab.field, root)
            continue
        df = root_corpus(root, relation, cache_dir, fetch)
        if not df.empty:
            by_prefix.setdefault(root.split(":", 1)[0].upper(), []).append((name, df))

    out: List[VocabCorpus] = []
    known = set()
    for prefix, parts in by_prefix.items():
        df = pd.concat([d for _, d in parts], ignore_index=True).drop_duplicates("obo_id")
        key = _safe("v_" + "_".join(r for r, _ in parts)
                    + ("_children" if vocab.root_property == "children" else ""))
        out.append(VocabCorpus(key=key, ontology_source=prefix.lower(),
                               corpus_df=df.reset_index(drop=True)))
        known |= set(df["label"].astype(str).str.strip().str.lower())

    code_of = dict(zip(vocab.allowed_values, vocab.allowed_codes))
    extra = [v for v in vocab.allowed_values if v.strip().lower() not in known]
    if extra:
        fkey = _safe(vocab.field)
        df = pd.DataFrame({
            "label": extra,
            "obo_id": [f"{LOCAL_CODE_PREFIX}:{fkey}_{i}" for i in range(len(extra))],
        })
        # Labels stay a local corpus even when the schema gives their CURIEs
        # (static.enum): those may span ontologies, and a closed list needs no
        # concept tables. The CURIEs are kept for the output term IDs.
        ids = {v.strip().lower(): code_of[v] for v in extra if code_of.get(v)}
        out.append(VocabCorpus(key=_safe("av_" + vocab.field), ontology_source="local",
                               corpus_df=df, local=True, term_ids=ids))
    return out


def build_field_corpora(vocabs: Mapping[str, FieldVocab],
                        cache_dir: Optional[Path] = None,
                        fetch: Optional[Callable[[str, str], List[dict]]] = None,
                        ) -> Dict[str, List[VocabCorpus]]:
    """Field → its corpora, for every field that has a vocabulary.

    Corpora with the same key (fields sharing roots) are the same object.
    """
    shared: Dict[str, VocabCorpus] = {}
    out: Dict[str, List[VocabCorpus]] = {}
    for f, v in vocabs.items():
        if not v.has_vocabulary:
            continue
        cs = [shared.setdefault(c.key, c) for c in build_corpora(v, cache_dir, fetch)]
        if cs:
            out[f] = cs
        else:
            logger.warning("[schema_vocab] %s: vocabulary resolved to no terms", f)
    return out


# --- drafting a schema table from a SchemaMapper preset ----------------------

# Starting vocabularies for the bundled SchemaMapper presets: curated field →
# (corpus_category, ontology_source), a key of OntoMapEngine's
# ``_CORPUS_REGISTRY``, whose root becomes the field's ``DynamicEnum``. Only
# :func:`draft_schema_table` reads this; :func:`~metaharmonizer.harmonize`
# takes every corpus from the schema table it is given.
PRESET_FIELD_CORPORA: Dict[str, Dict[str, Tuple[str, str]]] = {
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

SCHEMA_TABLE_COLUMNS = ["ColName", "ColClass", "MultipleValues", "DynamicEnum",
                        "DynamicEnumProperty", "AllowedValues"]


def draft_schema_table(schema: str, allowed_values: bool = True) -> pd.DataFrame:
    """A schema table drafted from a bundled SchemaMapper preset, to review.

    One row per field of the preset's target schema, in the layout
    :func:`read_schema_table` reads:

    - ``ColClass``: ``numeric`` for the preset's numeric fields, else
      ``character``.
    - ``DynamicEnum`` / ``DynamicEnumProperty``: for the fields in
      :data:`PRESET_FIELD_CORPORA`, the root of OntoMapEngine's default
      corpus for that field (e.g. ``NCIT:C3262`` for ``cancer_type``),
      ``descendant``.
    - ``AllowedValues`` (``allowed_values=True``): the preset's value
      dictionary, ``;``-joined. Fields with a label that contains ``;`` or
      looks like a pattern are left empty (the cell could not be read back);
      SchemaMapper still gets those values from the copied value dictionary
      (see :func:`write_schema_table_draft`).
    - ``MultipleValues``: empty; fill in ``TRUE`` for fields that hold
      several values per sample.

    The draft is a starting point. The roots are broad defaults (all of NCIt
    Neoplasm for every disease field), not a curated choice; narrow or
    replace them before relying on the result.
    """
    import json

    from metaharmonizer.engine.ontology_mapping_engine import _CORPUS_REGISTRY
    from metaharmonizer.models.schema_mapper.config import resolve_schema_preset

    preset = resolve_schema_preset(schema)
    ts = pd.read_csv(preset["target_schema_path"], dtype=str, keep_default_na=False)
    corpora = PRESET_FIELD_CORPORA.get(schema, {})
    values: Dict[str, list] = {}
    if allowed_values and Path(preset["value_dict_path"]).exists():
        with open(preset["value_dict_path"]) as fh:
            values = json.load(fh)

    rows, unreadable = [], []
    for f, num in zip(ts["field_name"], ts.get("is_numeric_field", [""] * len(ts))):
        spec = corpora.get(f)
        root = _CORPUS_REGISTRY[spec] if spec else ""
        labels = [str(v).strip() for v in values.get(f, []) if str(v).strip()]
        if any(";" in v or "|" in v or REGEX_SIGNS.search(v) for v in labels):
            unreadable.append(f)
            labels = []
        rows.append({
            "ColName": f,
            "ColClass": "numeric" if str(num).strip().lower() == "yes" else "character",
            "MultipleValues": "",
            "DynamicEnum": root,
            "DynamicEnumProperty": "descendant" if root else "",
            "AllowedValues": ";".join(dict.fromkeys(labels)),
        })
    if unreadable:
        logger.warning("[schema_vocab] %s: AllowedValues left empty for %d field(s) whose "
                       "labels contain ';' or look like patterns: %s",
                       schema, len(unreadable), unreadable[:5])
    return pd.DataFrame(rows, columns=SCHEMA_TABLE_COLUMNS)


def write_schema_table_draft(schema: str, path: Union[str, Path],
                             allowed_values: bool = True,
                             overwrite: bool = False) -> Path:
    """Write :func:`draft_schema_table` to ``path`` with the preset's dictionaries.

    The preset's SchemaMapper alias and value dictionaries are copied next to
    it as ``<stem>.alias.csv`` and ``<stem>.values.json``, which
    :func:`~metaharmonizer.harmonize` picks up, so schema mapping keeps the
    preset's aliases. Edit the CSV, then pass its path as ``schema_table``.
    """
    import shutil

    from metaharmonizer.models.schema_mapper.config import resolve_schema_preset

    path = Path(path)
    sibs = {".alias.csv": "alias_dict_path", ".values.json": "value_dict_path"}
    outs = [path] + [path.with_name(path.stem + s) for s in sibs]
    if not overwrite and any(p.exists() for p in outs):
        raise FileExistsError(f"{[str(p) for p in outs if p.exists()]} exist; "
                              f"pass overwrite=True to replace them")
    path.parent.mkdir(parents=True, exist_ok=True)
    draft_schema_table(schema, allowed_values).to_csv(path, index=False)
    preset = resolve_schema_preset(schema)
    for suffix, key in sibs.items():
        src = Path(preset[key])
        if src.exists():
            shutil.copyfile(src, path.with_name(path.stem + suffix))
    logger.info("[schema_vocab] wrote %s draft schema table to %s", schema, path)
    return path
