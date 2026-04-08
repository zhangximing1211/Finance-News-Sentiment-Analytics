from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"[.,;:]+$", "", value.strip())
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
