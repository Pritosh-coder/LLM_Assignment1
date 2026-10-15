"""Canonical Part A label taxonomy and cross-system normalization.

The canonical labels are the complete set observed in ``partA_train.jsonl``. Alias
tables intentionally contain only semantically defensible mappings. Ambiguous labels
such as ``PERSON`` and ``LOCATION`` return ``None`` rather than being forced into an
incorrect fine-grained class.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

CANONICAL_LABELS: tuple[str, ...] = (
    "BOD",
    "BUILDING",
    "CARDISSUER",
    "CITY",
    "COUNTRY",
    "DATE",
    "DRIVERLICENSE",
    "EMAIL",
    "GEOCOORD",
    "GIVENNAME1",
    "GIVENNAME2",
    "IDCARD",
    "IP",
    "LASTNAME1",
    "LASTNAME2",
    "LASTNAME3",
    "PASS",
    "PASSPORT",
    "POSTCODE",
    "SECADDRESS",
    "SEX",
    "SOCIALNUMBER",
    "STATE",
    "STREET",
    "TEL",
    "TIME",
    "TITLE",
    "USERNAME",
)
CANONICAL_LABEL_SET = frozenset(CANONICAL_LABELS)

# Keys are compacted to uppercase ASCII alphanumerics by _clean_label_token().
MODEL_LABEL_ALIASES: dict[str, str] = {
    "ACCOUNTNAME": "USERNAME",
    "BIRTHDATE": "BOD",
    "DATEOFBIRTH": "BOD",
    "DATETIME": "DATE",
    "DOB": "BOD",
    "DRIVERSLICENSE": "DRIVERLICENSE",
    "DRIVINGLICENSE": "DRIVERLICENSE",
    "EMAILADDRESS": "EMAIL",
    "FIRSTNAME": "GIVENNAME1",
    "GENDER": "SEX",
    "GEOCOORDINATE": "GEOCOORD",
    "GPSCOORDINATE": "GEOCOORD",
    "HONORIFIC": "TITLE",
    "IDCARDNUMBER": "IDCARD",
    "IDENTITYCARD": "IDCARD",
    "IPADDRESS": "IP",
    "IPV4": "IP",
    "IPV6": "IP",
    "LASTNAME": "LASTNAME1",
    "MIDDLENAME": "GIVENNAME2",
    "MOBILE": "TEL",
    "NEARBYGPSCOORDINATE": "GEOCOORD",
    "PASSCODE": "PASS",
    "PASSWORD": "PASS",
    "PHONE": "TEL",
    "PHONENUMBER": "TEL",
    "POSTALCODE": "POSTCODE",
    "PREFIX": "TITLE",
    "PROVINCE": "STATE",
    "SECONDARYADDRESS": "SECADDRESS",
    "SOCIALSECURITYNUMBER": "SOCIALNUMBER",
    "SSN": "SOCIALNUMBER",
    "STATEPROVINCE": "STATE",
    "SURNAME": "LASTNAME1",
    "TELEPHONE": "TEL",
    "ZIP": "POSTCODE",
    "ZIPCODE": "POSTCODE",
}

REGEX_LABEL_ALIASES: dict[str, str] = {
    "DATE": "DATE",
    "EMAILADDRESS": "EMAIL",
    "EMAIL": "EMAIL",
    "IPADDRESS": "IP",
    "IPV4": "IP",
    "IPV6": "IP",
    "PHONE": "TEL",
    "PHONENUMBER": "TEL",
    "TEL": "TEL",
    "TIME": "TIME",
}

PRESIDIO_LABEL_ALIASES: dict[str, str] = {
    "DATE_TIME": "DATE",
    "EMAIL_ADDRESS": "EMAIL",
    "IP_ADDRESS": "IP",
    "PHONE_NUMBER": "TEL",
    "US_DRIVER_LICENSE": "DRIVERLICENSE",
    "US_PASSPORT": "PASSPORT",
    "US_SSN": "SOCIALNUMBER",
}

_INSTANCE_SUFFIX_RE = re.compile(r"^(.*?)[_-](\d+)$")
_VALID_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]*$")
_ORDINAL_NAME_BASES: dict[str, str] = {
    "FIRSTNAME": "GIVENNAME",
    "GIVENNAME": "GIVENNAME",
    "LASTNAME": "LASTNAME",
    "SURNAME": "LASTNAME",
}


def _unwrap_label(label: str) -> str | None:
    value = label.strip()
    if value.startswith("[") or value.endswith("]"):
        if not (value.startswith("[") and value.endswith("]")):
            return None
        value = value[1:-1].strip()
    if not value or not _VALID_LABEL_RE.fullmatch(value):
        return None
    return value


def _compact_label_token(label: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", label.upper())


def _normalize_with_aliases(
    label: str,
    aliases: Mapping[str, str],
    *,
    allow_unknown: bool,
) -> str | None:
    unwrapped = _unwrap_label(label)
    if unwrapped is None:
        return None

    compact = _compact_label_token(unwrapped)
    if compact in CANONICAL_LABEL_SET:
        return compact
    if compact in aliases:
        return aliases[compact]

    # Model placeholders commonly add an instance suffix, e.g. [EMAIL_1]. Name
    # ordinals are semantic in this dataset, so [GIVENNAME_2] maps to GIVENNAME2
    # while [EMAIL_2] remains EMAIL.
    suffix_match = _INSTANCE_SUFFIX_RE.fullmatch(unwrapped.upper())
    if suffix_match:
        base = _compact_label_token(suffix_match.group(1))
        ordinal = int(suffix_match.group(2))
        name_stem = _ORDINAL_NAME_BASES.get(base)
        if name_stem is not None:
            ordinal_candidate = f"{name_stem}{ordinal}"
            if ordinal_candidate in CANONICAL_LABEL_SET:
                return ordinal_candidate
        if base in CANONICAL_LABEL_SET:
            return base
        if base in aliases:
            return aliases[base]

    return compact if allow_unknown else None


def normalize_gold_label(label: str, *, allow_unknown: bool = False) -> str | None:
    """Normalize gold labels, including declared aliases on OOD evaluation data.

    Training-taxonomy derivation uses the strict default. Evaluation calls with
    ``allow_unknown=True``: known OOD synonyms are then mapped to the same canonical
    labels as system outputs, while genuinely novel labels remain distinct.
    """
    aliases = MODEL_LABEL_ALIASES if allow_unknown else {}
    return _normalize_with_aliases(label, aliases, allow_unknown=allow_unknown)


def normalize_model_label(label: str, *, allow_unknown: bool = False) -> str | None:
    """Normalize a model-emitted placeholder name."""
    return _normalize_with_aliases(
        label, MODEL_LABEL_ALIASES, allow_unknown=allow_unknown
    )


def normalize_regex_label(label: str, *, allow_unknown: bool = False) -> str | None:
    """Map a regex detector category into the canonical taxonomy."""
    aliases = {**MODEL_LABEL_ALIASES, **REGEX_LABEL_ALIASES}
    return _normalize_with_aliases(
        label, aliases, allow_unknown=allow_unknown
    )


def normalize_presidio_label(label: str, *, allow_unknown: bool = False) -> str | None:
    """Map a Presidio entity type into the canonical taxonomy."""
    unwrapped = _unwrap_label(label)
    if unwrapped is None:
        return None
    mapped = PRESIDIO_LABEL_ALIASES.get(unwrapped.upper())
    if mapped is not None:
        return mapped
    return _normalize_with_aliases(
        unwrapped, MODEL_LABEL_ALIASES, allow_unknown=allow_unknown
    )


def canonical_label_frequencies(
    records: Iterable[Mapping[str, Any]],
) -> Counter[str]:
    """Count canonical gold labels, rejecting malformed or unknown training labels."""
    frequencies: Counter[str] = Counter()
    for row_index, record in enumerate(records, start=1):
        masks = record.get("privacy_mask")
        if not isinstance(masks, list):
            raise ValueError(f"record {row_index}: privacy_mask must be a list")
        for mask_index, mask in enumerate(masks, start=1):
            if not isinstance(mask, Mapping):
                raise ValueError(
                    f"record {row_index}, mask {mask_index}: mask must be an object"
                )
            raw_label = mask.get("label")
            if not isinstance(raw_label, str):
                raise ValueError(
                    f"record {row_index}, mask {mask_index}: label must be a string"
                )
            canonical = normalize_gold_label(raw_label)
            if canonical is None:
                raise ValueError(
                    f"record {row_index}, mask {mask_index}: unknown label {raw_label!r}"
                )
            frequencies[canonical] += 1
    return frequencies


def top_canonical_labels(
    records: Iterable[Mapping[str, Any]], limit: int = 10
) -> list[tuple[str, int]]:
    """Return the most frequent labels with deterministic tie-breaking."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    frequencies = canonical_label_frequencies(records)
    return sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:limit]
