from __future__ import annotations

SYNONYM_MAP: dict[str, list[str]] = {
    "domestic worker": [
        "domestic helper",
        "foreign domestic helper",
    ],
    "domestic workers": [
        "domestic helpers",
        "foreign domestic helpers",
    ],
    "fdh": [
        "foreign domestic helper",
        "domestic helper",
    ],
    "agency": [
        "employment agency",
    ],
    "agencies": [
        "employment agencies",
    ],
    "leave": [
        "annual leave",
        "statutory holidays",
    ],
    "wages": [
        "salary",
        "payment of wages",
    ],
    "termination": [
        "dismissal",
        "notice period",
    ],
}


def expand_query(query: str) -> list[str]:
    query = query.strip()
    if not query:
        return []

    expansions = [query]
    lowered = query.lower()

    for term, synonyms in SYNONYM_MAP.items():
        if term in lowered:
            for synonym in synonyms:
                candidate = lowered.replace(term, synonym)
                if candidate not in expansions:
                    expansions.append(candidate)

    # small heuristic expansion for legal phrasing
    if "rights" in lowered and "employment rights" not in expansions:
        expansions.append("employment rights")

    return expansions