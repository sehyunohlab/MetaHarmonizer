"""Read the value columns of one schema-table row.

Shared by :mod:`metaharmonizer.schema_vocab` (``harmonize()``) and
:class:`~metaharmonizer.knowledge_db.combined_corpus_builder.CombinedCorpusBuilder`
so both read a row the same way. Needs only pandas, so the schema registry's CI
can import it without MetaHarmonizer's heavy dependencies.

The registry convention (MetaHarmonizerSchemaRegistry ``docs/target_schema.md``):

- ``corpus.type``: ``|``-joined tokens from ``dynamic_enum``, ``static_enum``,
  ``custom_enum``, ``regexp`` and ``any`` (``;`` is accepted too). It decides
  how ``allowedvalues`` is read: a ``static_enum`` or ``custom_enum`` row holds a
  ``|`` list of labels, any other row holds no labels.
- ``static.enum`` (``static_enum`` rows only): one CURIE per label, same order.
  A label without an ID has an empty slot (``NCIT:C1||NCIT:C3``) or ``NA``.
  With no labels, the CURIEs are standalone terms whose labels come from the
  ontology (``Healthy`` for ``disease``: ``NCIT:C115935``).

Rows whose ``corpus.type`` is empty or holds only retired tokens (``binary``
aside, which is read as ``static_enum``) are read the legacy way: labels are
split on ``legacy_sep`` and a cell that looks like a regular expression is a
pattern, not a list.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CORPUS_TYPES = frozenset({"dynamic_enum", "static_enum", "custom_enum", "regexp", "any"})
LIST_TYPES = frozenset({"static_enum", "custom_enum"})
# Retired tokens still read: binary was a Yes|No (or TRUE|FALSE) static list.
_RETIRED_AS = {"binary": "static_enum"}
_NULLS = {"", "na", "nan", "none", "null"}

# Signs that a legacy allowedvalues item is a pattern, not a label: a character
# class, an escape, anchors, a counted repeat, ``.+``/``.*`` or an optional
# group. Plain ``*``, ``+`` or ``(...)`` occur in real labels
# (``HLA-DQA1*02:01 protein complex``, ``CD4+ T cell``, ``smoker (finding)``).
REGEX_SIGNS = re.compile(r"\[[^\]]*\]|\\[a-zA-Z.]|^\^|\$$|\{\d+(,\d*)?\}|\.[+*]|\)\?")


def cell(v) -> str:
    """A cell as a stripped string; ``""`` for None, NaN and NA-like text."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    v = str(v).strip()
    return "" if v.lower() in _NULLS else v


def split_list(v, keep_empty: bool = False) -> Tuple[str, ...]:
    """Items of a ``|`` list. ``keep_empty`` keeps empty and ``NA`` slots as
    ``""`` so the items still pair by position."""
    s = cell(v)
    if not s:
        return ()
    items = tuple(cell(t) for t in s.split("|"))
    return items if keep_empty else tuple(t for t in items if t)


def corpus_types(v) -> FrozenSet[str]:
    """The known tokens of a ``corpus.type`` cell (``binary`` read as
    ``static_enum``); empty when there are none."""
    toks = (t.strip() for t in re.split(r"[|;]", cell(v).lower()))
    return frozenset(_RETIRED_AS.get(t, t) for t in toks) & CORPUS_TYPES


@dataclass(frozen=True)
class RowValues:
    """The labels and term IDs a row declares."""

    labels: Tuple[str, ...] = ()      # closed-list labels, de-duplicated
    codes: Tuple[str, ...] = ()       # CURIE per label, "" where none
    static_terms: Tuple[str, ...] = ()  # standalone CURIEs (no labels)
    typed: bool = False               # read by corpus.type, not the legacy way


def _legacy_labels(v, sep: str) -> Tuple[str, ...]:
    s = cell(v)
    if not s:
        return ()
    items = [t.strip() for t in s.split(sep) if t.strip()]
    if any(REGEX_SIGNS.search(t) for t in items):
        return ()
    # In a ;-separated cell, | was regex alternation.
    if sep != "|" and any("|" in t for t in items):
        return ()
    return tuple(t for t in items if cell(t))


def _pair(labels, codes, f: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if codes and len(codes) != len(labels):
        logger.warning("[schema_cells] %s: %d static.enum codes for %d labels; codes ignored "
                       "(give a label without an ID an empty slot, e.g. NCIT:C1||NCIT:C3)",
                       f, len(codes), len(labels))
        codes = ()
    codes = codes or ("",) * len(labels)
    pairs: dict = {}
    for lab, code in zip(labels, codes):
        if lab and lab not in pairs:  # de-duplicate labels, keep first code
            pairs[lab] = code
    return tuple(pairs), tuple(pairs.values())


def read_row_values(corpus_type, allowedvalues, static_enum, field: str = "",
                    legacy_sep: str = ";") -> RowValues:
    """Labels, their term IDs and standalone terms of one schema row."""
    types = corpus_types(corpus_type)
    if not types:
        labels = _legacy_labels(allowedvalues, legacy_sep)
        codes = split_list(static_enum, keep_empty=True)
        if labels:
            # Legacy rows pair IDs only when the counts line up.
            if len(codes) != len(labels):
                codes = ()
            labels, codes = _pair(labels, codes, field)
            return RowValues(labels, codes)
        return RowValues(static_terms=tuple(c for c in codes if c))

    labels: Tuple[str, ...] = ()
    if types & LIST_TYPES:
        labels = split_list(allowedvalues, keep_empty=True)
    codes = split_list(static_enum, keep_empty=True) if "static_enum" in types else ()
    if any(labels):
        labels, codes = _pair(labels, codes, field)
        return RowValues(labels, codes, typed=True)
    return RowValues(static_terms=tuple(dict.fromkeys(c for c in codes if c)), typed=True)
