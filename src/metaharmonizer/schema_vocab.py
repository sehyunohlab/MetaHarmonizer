"""Per-field value vocabularies read from a schema table.

A schema table (a *data dictionary*) lists the curated fields and, for each,
the vocabulary its values come from. Two columns carry that vocabulary:

- ``DynamicEnum`` (alias ``ontology_root``): one or more ontology root CURIEs,
  ``;`` or ``|`` separated, e.g. ``NCIT:C7057;EFO:0000408``. With
  ``DynamicEnumProperty`` = ``descendant`` (default) the vocabulary is every
  descendant of each root plus the root; with ``children`` it is the direct
  children only.
- ``AllowedValues``: a ``;``-separated list of permitted labels, e.g.
  ``Female;Male``. A value that is a regular expression (``[0-9]+``, ``.+``)
  describes a format, not a vocabulary, and is ignored.

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

logger = logging.getLogger(__name__)

# Column aliases: data-dictionary name first, MetaHarmonizer name second.
_NAME_COLS = ("ColName", "field_name")
_ROOT_COLS = ("DynamicEnum", "ontology_root")
_PROP_COLS = ("DynamicEnumProperty", "ontology_root_property")
_ALLOWED_COLS = ("AllowedValues", "allowed_values")
_MULTI_COLS = ("MultipleValues", "multiple_values")
_CLASS_COLS = ("ColClass",)
_NUMERIC_CLASSES = {"integer", "numeric", "double", "float", "number"}

# Prefix of the codes given to allowed-value labels that have no ontology
# code. OntoMapEngine treats a corpus whose code prefix it cannot resolve as a
# local corpus (no concept tables); harmonize() leaves their term_id empty.
LOCAL_CODE_PREFIX = "LOCAL"

# Signs that an AllowedValues item is a pattern, not a label: a character
# class, an escape, anchors, a counted repeat, ``.+``/``.*``, an optional
# group, or alternation. Plain ``*`` or ``(...)`` occur in real labels
# (``HLA-DQA1*02:01 protein complex``, ``smoker (finding)``).
_REGEX_SIGNS = re.compile(r"\[[^\]]*\]|\\[a-zA-Z.]|^\^|\$$|\{\d+(,\d*)?\}|\.[+*]|\)\?|\|")
_NULLS = {"", "na", "nan", "none", "null"}


@dataclass(frozen=True)
class FieldVocab:
    """Vocabulary of one curated field."""

    field: str
    roots: Tuple[str, ...] = ()
    root_property: str = "descendant"
    allowed_values: Tuple[str, ...] = ()
    multiple_values: bool = False

    @property
    def has_vocabulary(self) -> bool:
        return bool(self.roots or self.allowed_values)


@dataclass
class VocabCorpus:
    """One OntoMapEngine corpus serving one or more fields."""

    key: str                    # safe identifier, used as corpus_category
    ontology_source: str        # e.g. "ncit", "hancestro", "local"
    corpus_df: pd.DataFrame = field(repr=False)
    local: bool = False         # codes are LOCAL_*, not real ontology codes


def _pick(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    return next((c for c in names if c in df.columns), None)


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    v = str(v).strip()
    return "" if v.lower() in _NULLS else v


def parse_allowed_values(cell) -> Tuple[str, ...]:
    """Labels from an ``AllowedValues`` cell, or ``()`` for a pattern.

    The whole cell is treated as a pattern (and ignored) when any item looks
    like a regular expression, e.g. ``[0-9]+|NA`` or ``.+``.
    """
    s = _cell(cell)
    if not s:
        return ()
    items = [t.strip() for t in s.split(";") if t.strip()]
    if any(_REGEX_SIGNS.search(t) for t in items):
        return ()
    return tuple(dict.fromkeys(items))


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
    out: Dict[str, FieldVocab] = {}
    for _, r in df.iterrows():
        f = _cell(r[name])
        if not f:
            continue
        prop = (_cell(r[prop_c]) if prop_c else "").lower() or "descendant"
        if prop not in {"descendant", "children"}:
            logger.warning("[schema_vocab] %s: unknown root property %r; using descendant", f, prop)
            prop = "descendant"
        out[f] = FieldVocab(
            field=f,
            roots=_parse_roots(r[root_c]) if root_c else (),
            root_property=prop,
            allowed_values=parse_allowed_values(r[allow_c]) if allow_c else (),
            multiple_values=_to_bool(r[multi_c]) if multi_c else False,
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
    it names the category, e.g. ``Menopause Status``, not a value). An
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

    One corpus per ontology prefix among the field's roots, keyed by the roots
    (so fields with the same roots share it), plus a local corpus for allowed
    values whose label is not already in those corpora.
    """
    by_prefix: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    for root in vocab.roots:
        if ":" not in root:
            logger.warning("[schema_vocab] %s: root %r is not a CURIE; skipped", vocab.field, root)
            continue
        df = root_corpus(root, vocab.root_property, cache_dir, fetch)
        if not df.empty:
            by_prefix.setdefault(root.split(":", 1)[0].upper(), []).append((root, df))

    out: List[VocabCorpus] = []
    known = set()
    for prefix, parts in by_prefix.items():
        df = pd.concat([d for _, d in parts], ignore_index=True).drop_duplicates("obo_id")
        key = _safe("v_" + "_".join(r for r, _ in parts)
                    + ("_children" if vocab.root_property == "children" else ""))
        out.append(VocabCorpus(key=key, ontology_source=prefix.lower(),
                               corpus_df=df.reset_index(drop=True)))
        known |= set(df["label"].astype(str).str.strip().str.lower())

    extra = [v for v in vocab.allowed_values if v.strip().lower() not in known]
    if extra:
        fkey = _safe(vocab.field)
        df = pd.DataFrame({
            "label": extra,
            "obo_id": [f"{LOCAL_CODE_PREFIX}:{fkey}_{i}" for i in range(len(extra))],
        })
        out.append(VocabCorpus(key=_safe("av_" + vocab.field), ontology_source="local",
                               corpus_df=df, local=True))
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
