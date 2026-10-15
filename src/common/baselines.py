"""Reproducible regex and Presidio sanitization baselines for Part B."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from src.common.labels import normalize_presidio_label, normalize_regex_label


class Detection(TypedDict):
    start: int
    end: int
    label: str
    rule: str
    score: float


class BaselineResult(TypedDict):
    predicted_text: str
    predicted_spans: list[Detection]
    well_formed: bool
    issues: list[dict[str, Any]]


class _PresidioResult(Protocol):
    start: int
    end: int
    entity_type: str
    score: float


class _Analyzer(Protocol):
    def analyze(self, *, text: str, language: str) -> Sequence[_PresidioResult]: ...


@dataclass(frozen=True)
class _RegexRule:
    name: str
    label: str
    pattern: re.Pattern[str]
    score: float
    group: str | int = 0
    validator: Callable[[str], bool] | None = None


_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
_NUMERIC_DATE = (
    r"(?:\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})|\d{4}-\d{2}-\d{2})"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?)?"
)
_TEXT_DATE = (
    rf"(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:\s+\d{{4}})?)"
)
_TIME = (
    r"(?:(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?[APap][Mm])?|"
    r"(?:1[0-2]|[1-9])\s?[APap][Mm])"
)
_CONTEXT_VALUE = r"\s*(?:is\s+|=\s*|:\s*|#\s*)?(?P<value>"


def _valid_ipv4(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).version == 4
    except ValueError:
        return False


def _valid_ipv6(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).version == 6
    except ValueError:
        return False


def _valid_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15:
        return False
    if re.fullmatch(_NUMERIC_DATE, value):
        return False
    return value.startswith("+") or bool(re.search(r"[ ().-]", value)) or len(digits) >= 10


def _valid_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _compile(pattern: str, *, ignore_case: bool = False) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pattern, flags)


_REGEX_RULES: tuple[_RegexRule, ...] = (
    _RegexRule(
        "email",
        "EMAIL_ADDRESS",
        _compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)"),
        0.995,
    ),
    _RegexRule(
        "url",
        "URL",
        _compile(
            r"(?i)(?<!\w)(?:https?://|www\.)[^\s<>\[\]\"']*"
            r"[A-Za-z0-9/#](?!\w)"
        ),
        0.99,
    ),
    _RegexRule(
        "geo_coordinate",
        "NEARBYGPSCOORDINATE",
        _compile(
            r"\[?[-+]?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?)\s*,\s*"
            r"[-+]?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|[0-9]?\d(?:\.\d+)?)\]?"
        ),
        0.985,
    ),
    _RegexRule(
        "ipv4",
        "IPV4",
        _compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
        0.98,
        validator=_valid_ipv4,
    ),
    _RegexRule(
        "ipv6",
        "IPV6",
        _compile(
            r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}"
            r"[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])"
        ),
        0.98,
        validator=_valid_ipv6,
    ),
    _RegexRule(
        "mac_address",
        "MAC",
        _compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])"),
        0.98,
    ),
    _RegexRule(
        "iban",
        "IBAN",
        _compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}(?![A-Z0-9])"),
        0.97,
    ),
    _RegexRule(
        "ethereum_address",
        "ETHEREUMADDRESS",
        _compile(r"(?<![A-Fa-f0-9])0x[A-Fa-f0-9]{40}(?![A-Fa-f0-9])"),
        0.97,
    ),
    _RegexRule(
        "bitcoin_address",
        "BITCOINADDRESS",
        _compile(
            r"(?<![A-Za-z0-9])(?:bc1[ac-hj-np-z02-9]{25,62}|"
            r"[13][A-HJ-NP-Za-km-z1-9]{25,34})(?![A-Za-z0-9])"
        ),
        0.97,
    ),
    _RegexRule(
        "litecoin_address",
        "LITECOINADDRESS",
        _compile(
            r"(?<![A-Za-z0-9])(?:ltc1[ac-hj-np-z02-9]{25,62}|"
            r"[LM3][A-HJ-NP-Za-km-z1-9]{25,34})(?![A-Za-z0-9])"
        ),
        0.97,
    ),
    _RegexRule(
        "credit_card_luhn",
        "CREDITCARDNUMBER",
        _compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
        0.965,
        validator=_valid_luhn,
    ),
    _RegexRule(
        "social_number",
        "SSN",
        _compile(r"(?<!\d)\d{3}[ .-]\d{2,3}[ .-]\d{4}(?!\d)"),
        0.96,
    ),
    _RegexRule(
        "dob_context",
        "DOB",
        _compile(
            rf"(?i)\b(?:dob|date\s+of\s+birth|birth\s+date|born)\b"
            rf"{_CONTEXT_VALUE}(?:{_NUMERIC_DATE}|{_TEXT_DATE}))"
        ),
        0.955,
        group="value",
    ),
    _RegexRule(
        "driver_license_context",
        "DRIVERLICENSE",
        _compile(
            rf"(?i)\b(?:driver'?s?\s+licen[cs]e|driving\s+licen[cs]e|dl)\b"
            rf"{_CONTEXT_VALUE}[A-Z0-9]{{1,8}}(?:[ .-][A-Z0-9]{{1,8}}){{0,3}})"
        ),
        0.95,
        group="value",
    ),
    _RegexRule(
        "passport_context",
        "PASSPORT",
        _compile(
            rf"(?i)\bpassport(?:\s+(?:number|no))?\b"
            rf"{_CONTEXT_VALUE}[A-Z0-9]{{2,18}})"
        ),
        0.95,
        group="value",
    ),
    _RegexRule(
        "id_card_context",
        "IDCARD",
        _compile(
            rf"(?i)\b(?:id(?:entity|entification)?\s*(?:card|number|no)?|"
            rf"identity\s+card)\b{_CONTEXT_VALUE}[A-Z0-9]{{2,20}})"
        ),
        0.95,
        group="value",
    ),
    _RegexRule(
        "imei_context",
        "PHONEIMEI",
        _compile(rf"(?i)\bimei\b{_CONTEXT_VALUE}\d{{15}})"),
        0.95,
        group="value",
    ),
    _RegexRule(
        "vin",
        "VEHICLEVIN",
        _compile(r"(?<![A-HJ-NPR-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-HJ-NPR-Z0-9])"),
        0.94,
    ),
    _RegexRule(
        "username_context",
        "ACCOUNTNAME",
        _compile(
            rf"(?i)\b(?:user\s*name|account\s*name|login|handle)\b"
            rf"{_CONTEXT_VALUE}[^\s,;]+)"
        ),
        0.93,
        group="value",
    ),
    _RegexRule(
        "password_context",
        "PASSWORD",
        _compile(
            rf"(?i)\b(?:password|passcode|pwd)\b{_CONTEXT_VALUE}[^\s,;]+)"
        ),
        0.93,
        group="value",
    ),
    _RegexRule(
        "cvv_context",
        "CREDITCARDCVV",
        _compile(rf"(?i)\b(?:cvv|cvc)\b{_CONTEXT_VALUE}\d{{3,4}})"),
        0.93,
        group="value",
    ),
    _RegexRule(
        "pin_context",
        "PIN",
        _compile(rf"(?i)\bpin\b{_CONTEXT_VALUE}\d{{3,8}})"),
        0.93,
        group="value",
    ),
    _RegexRule(
        "postcode_context",
        "ZIPCODE",
        _compile(
            rf"(?i)\b(?:post(?:al)?\s*code|zip(?:\s*code)?)\b"
            rf"{_CONTEXT_VALUE}[A-Z0-9]{{3,5}}(?:[ -][A-Z0-9]{{2,4}})?)"
        ),
        0.92,
        group="value",
    ),
    _RegexRule(
        "uk_postcode",
        "ZIPCODE",
        _compile(r"(?i)(?<![A-Z0-9])[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}(?![A-Z0-9])"),
        0.91,
    ),
    _RegexRule(
        "date_numeric",
        "DATE",
        _compile(rf"(?<!\d){_NUMERIC_DATE}(?!\d)"),
        0.88,
    ),
    _RegexRule(
        "date_text",
        "DATE",
        _compile(rf"(?i)(?<!\w){_TEXT_DATE}(?!\w)"),
        0.88,
    ),
    _RegexRule(
        "time",
        "TIME",
        _compile(rf"(?<!\w){_TIME}(?!\w)"),
        0.86,
    ),
    _RegexRule(
        "phone",
        "PHONE_NUMBER",
        _compile(r"(?<![\w])\+?\d(?:[\d ().-]{5,}\d)(?![\w])"),
        0.75,
        validator=_valid_phone,
    ),
)


def _select_non_overlapping(detections: Sequence[Detection]) -> list[Detection]:
    selected: list[Detection] = []
    for detection in sorted(
        detections,
        key=lambda item: (
            -item["score"],
            -(item["end"] - item["start"]),
            item["start"],
            item["end"],
            item["label"],
        ),
    ):
        if any(
            detection["start"] < existing["end"]
            and existing["start"] < detection["end"]
            for existing in selected
        ):
            continue
        selected.append(detection)
    return sorted(selected, key=lambda item: (item["start"], item["end"]))


def _replace_detections(text: str, detections: Sequence[Detection]) -> str:
    output = text
    for detection in reversed(detections):
        output = (
            output[: detection["start"]]
            + f"[{detection['label']}]"
            + output[detection["end"] :]
        )
    return output


def regex_sanitize(text: str) -> BaselineResult:
    """Apply documented, context-independent regex rules to one string."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    candidates: list[Detection] = []
    for rule in _REGEX_RULES:
        for match in rule.pattern.finditer(text):
            start, end = match.span(rule.group)
            value = text[start:end]
            if rule.validator is not None and not rule.validator(value):
                continue
            label = normalize_regex_label(rule.label, allow_unknown=True)
            if label is None:
                continue
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "rule": f"regex:{rule.name}",
                    "score": rule.score,
                }
            )
    selected = _select_non_overlapping(candidates)
    return {
        "predicted_text": _replace_detections(text, selected),
        "predicted_spans": selected,
        "well_formed": True,
        "issues": [],
    }


def build_default_presidio_analyzer() -> _Analyzer:
    """Construct Presidio's default English analyzer lazily."""
    try:
        import tldextract
        from presidio_analyzer import AnalyzerEngine
    except ImportError as error:  # pragma: no cover - depends on optional baseline
        raise RuntimeError(
            "Presidio baseline requires presidio-analyzer and its English NLP model"
        ) from error
    # Presidio's stock EmailRecognizer calls the module-level tldextract helper.
    # Freeze it to tldextract's bundled PSL snapshot: no network refresh, no mutable
    # user cache, and identical recognition rules across grading machines.
    tldextract.extract = tldextract.TLDExtract(
        suffix_list_urls=(), cache_dir=None, fallback_to_snapshot=True
    )
    try:
        return AnalyzerEngine()
    except Exception as error:  # pragma: no cover - external NLP model setup
        raise RuntimeError(
            "Presidio could not initialize its default English NLP engine; install "
            "the model declared by the pinned Presidio release"
        ) from error


def presidio_sanitize(text: str, analyzer: _Analyzer | None = None) -> BaselineResult:
    """Sanitize using native Presidio spans and the shared label taxonomy."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    active_analyzer = analyzer if analyzer is not None else build_default_presidio_analyzer()
    candidates: list[Detection] = []
    for result in active_analyzer.analyze(text=text, language="en"):
        if not 0 <= result.start < result.end <= len(text):
            continue
        label = normalize_presidio_label(result.entity_type, allow_unknown=True)
        if label is None:
            continue
        candidates.append(
            {
                "start": result.start,
                "end": result.end,
                "label": label,
                "rule": f"presidio:{result.entity_type}",
                "score": float(result.score),
            }
        )
    selected = _select_non_overlapping(candidates)
    return {
        "predicted_text": _replace_detections(text, selected),
        "predicted_spans": selected,
        "well_formed": True,
        "issues": [],
    }
