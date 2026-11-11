"""Recover questions from an imperfect Clean DSG generation.

This is the ``greedy_question_salvage_v1`` parser used for the released
SVGFaith-Score. It keeps every nonempty question, drops exact duplicates, and
remaps parent ids when each original id survives once. It does not check
types, subtypes, evidence, or graph semantics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterator


_QUESTION_STRING_RE = re.compile(
    r'"question"\s*:\s*("(?:\\.|[^"\\])*")', re.DOTALL
)


@dataclass(frozen=True)
class SalvageResult:
    decomposition: dict[str, Any]
    repairs: list[dict[str, Any]]
    discarded: list[dict[str, Any]]
    parse_sources: list[str]


def _walk_question_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if "question" in value:
            yield value
        for child in value.values():
            yield from _walk_question_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_question_dicts(child)


def _decoded_fragments(content: str) -> list[tuple[Any, str]]:
    decoder = json.JSONDecoder()
    fragments: list[tuple[Any, str]] = []
    seen_spans: set[tuple[int, int]] = set()
    stripped = content.strip()
    try:
        fragments.append((json.loads(stripped), "complete_json"))
        return fragments
    except (json.JSONDecodeError, TypeError):
        pass
    for start, char in enumerate(content):
        if char not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            continue
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        fragments.append((value, "embedded_json_fragment"))
    return fragments


def salvage_questions(content: str) -> SalvageResult:
    """Parse ``content`` and return tuples whose ids are renumbered from 1."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("assistant final content is empty")

    candidates: list[tuple[dict[str, Any], str]] = []
    parse_sources: list[str] = []
    decoded = _decoded_fragments(content)
    for value, source in decoded:
        parse_sources.append(source)
        candidates.extend((item, source) for item in _walk_question_dicts(value))

    if not decoded or "complete_json" not in parse_sources:
        for match in _QUESTION_STRING_RE.finditer(content):
            try:
                question = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            candidates.append(({"question": question}, "question_string_scan"))
        if _QUESTION_STRING_RE.search(content):
            parse_sources.append("question_string_scan")

    tuples: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    for position, (candidate, source) in enumerate(candidates, 1):
        question_value = candidate.get("question")
        if not isinstance(question_value, str) or not question_value.strip():
            discarded.append({
                "candidate": position,
                "reason": "question_missing_or_empty",
                "source": source,
            })
            continue
        question = question_value.strip()
        if question in seen_questions:
            discarded.append({
                "candidate": position,
                "question": question,
                "reason": "exact_duplicate_question",
                "source": source,
            })
            continue
        seen_questions.add(question)

        original_id = candidate.get("id")
        atom = dict(candidate)
        atom["id"] = len(tuples) + 1
        atom["question"] = question
        defaults: list[str] = []
        if not isinstance(atom.get("type"), str) or not atom["type"].strip():
            atom["type"] = "other"
            defaults.append("type=other")
        if not isinstance(atom.get("subtype"), str) or not atom["subtype"].strip():
            atom["subtype"] = "other"
            defaults.append("subtype=other")
        if not isinstance(atom.get("args"), list):
            atom["args"] = []
            defaults.append("args=[]")
        if not isinstance(atom.get("parents"), list):
            atom["parents"] = []
            defaults.append("parents=[]")
        if not isinstance(atom.get("evidence"), str):
            atom["evidence"] = ""
            defaults.append("evidence='' ")
        if atom.get("polarity") not in {"present", "absent"}:
            atom["polarity"] = "present"
            defaults.append("polarity=present")
        repairs.append({
            "candidate": position,
            "new_id": atom["id"],
            "original_id": original_id,
            "question": question,
            "source": source,
            "defaults": defaults,
            "type_validation_performed": False,
        })
        tuples.append(atom)

    if not tuples:
        raise ValueError("no nonempty question could be recovered")

    original_id_counts: dict[int, int] = {}
    for repair in repairs:
        original_id = repair["original_id"]
        if isinstance(original_id, int) and not isinstance(original_id, bool):
            original_id_counts[original_id] = original_id_counts.get(original_id, 0) + 1
    id_map = {
        repair["original_id"]: repair["new_id"]
        for repair in repairs
        if isinstance(repair["original_id"], int)
        and not isinstance(repair["original_id"], bool)
        and original_id_counts[repair["original_id"]] == 1
    }
    for atom, repair in zip(tuples, repairs):
        old_parents = atom["parents"]
        remapped = [
            id_map[parent]
            for parent in old_parents
            if isinstance(parent, int)
            and not isinstance(parent, bool)
            and parent in id_map
            and id_map[parent] != atom["id"]
        ]
        remapped = list(dict.fromkeys(remapped))
        if remapped != old_parents:
            atom["parents"] = remapped
            repair["parent_repair"] = {
                "original": old_parents,
                "remapped": remapped,
                "policy": "retain_only_unambiguous_recovered_parent_ids",
            }

    return SalvageResult(
        decomposition={"tuples": tuples},
        repairs=repairs,
        discarded=discarded,
        parse_sources=list(dict.fromkeys(parse_sources)),
    )
