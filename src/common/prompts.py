"""Frozen M1 sanitizer prompt contract.

Changing this after the M1 gate would invalidate later comparisons and the Part C
sanitized-corpus lineage.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Sanitize the user's text by replacing every personally identifiable information "
    "value with its bracketed PII label. Preserve all non-PII text exactly. Return only "
    "the sanitized text, with no explanation."
)


def build_prompt_messages(source_text: str) -> list[dict[str, str]]:
    """Return the system/user messages used for both training and inference."""
    if not isinstance(source_text, str) or not source_text:
        raise ValueError("source_text must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source_text},
    ]
