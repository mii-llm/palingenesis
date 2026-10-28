"""PII filter: runs on every note before anything is written to the store.

Two categories, with a policy per scope level:

    category   examples                                          user scope           team / org scope
    secret     API keys, private keys, passwords, card numbers   rejected             rejected
               (Luhn), IBANs (mod-97 checksum)
    personal   emails, phone numbers, Italian codice fiscale     allow (default)      redact (default)

`allow` stores the text as is (a user's own memory; `forget` removes it), `redact` replaces each match
with a placeholder such as [EMAIL], `block` holds the note in the inbox for review. Detectors are
pluggable: RegexDetector is the baseline, and any object with `find(text) -> list[Finding]` (a wrapper
around Presidio or a judge model, for example) can be added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

SECRET, PERSONAL = "secret", "personal"


@dataclass(frozen=True)
class Finding:
    category: str  # secret | personal
    kind: str  # email, phone, iban, codice_fiscale, credit_card, api_key, private_key, password
    start: int
    end: int


class Detector(Protocol):
    def find(self, text: str) -> list[Finding]: ...


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(iban: str) -> bool:
    """ISO 13616 mod-97 check."""
    iban = iban.replace(" ", "").upper()
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    try:
        return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1
    except ValueError:
        return False


_CF_ODD = {
    **dict(zip("0123456789", [1, 0, 5, 7, 9, 13, 15, 17, 19, 21])),
    **dict(
        zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23],
        )
    ),
}


def codice_fiscale_ok(code: str) -> bool:
    """The control character of an Italian codice fiscale (positions counted from 1: odd and even tables)."""
    code = code.upper()
    if len(code) != 16:
        return False
    total = 0
    for i, ch in enumerate(code[:15]):
        if i % 2 == 0:
            total += _CF_ODD[ch]
        else:
            total += int(ch) if ch.isdigit() else ord(ch) - ord("A")
    return chr(ord("A") + total % 26) == code[15]


# Omocodia replaces digits of the code with letters (LMNPQRSTUV), so digit positions accept those too.
_D = "[0-9LMNPQRSTUV]"
_PATTERNS = [
    (PERSONAL, "email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")),
    (PERSONAL, "codice_fiscale", re.compile(rf"\b[A-Za-z]{{6}}{_D}{{2}}[A-Za-z]{_D}{{2}}[A-Za-z]{_D}{{3}}[A-Za-z]\b")),
    (SECRET, "iban", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")),
    (SECRET, "credit_card", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    # International numbers (+39 ..., 0039 ...), Italian mobile (3xx) and landline (0x) numbers, and the
    # (555) 123-4567 form. Bare digit runs are not phones: dates, versions and amounts must survive.
    (PERSONAL, "phone", re.compile(r"(?<![\w+])(?:\+|00)[1-9]\d{0,2}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?!\w)")),
    (PERSONAL, "phone", re.compile(r"(?<![\w.-])(?:3\d{2}[ .-]?\d{3}[ .-]?\d{3,4}|0\d{1,3}[ .-]?\d{5,8})(?![\w.-])")),
    (PERSONAL, "phone", re.compile(r"\(\d{3}\)[ ]?\d{3}-\d{4}\b")),
    (
        SECRET,
        "api_key",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk_(?:live|test)_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}"
            r"|hf_[A-Za-z0-9]{30,}|xox[abpr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b"
        ),
    ),
    (SECRET, "private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    (SECRET, "password", re.compile(r"(?i)\b(?:password|passwd|pwd|parola d'ordine)\s*[:=]\s*\S+")),
]


class RegexDetector:
    """Regex baseline with checksums: emails, phone numbers, codice fiscale, IBAN, card numbers, keys."""

    def find(self, text: str) -> list[Finding]:
        found: list[Finding] = []
        taken: list[tuple[int, int]] = []
        for category, kind, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                start, end = m.span()
                if any(a < end and start < b for a, b in taken):
                    continue  # a span already claimed by a more specific detector
                value = m.group(0)
                digits = re.sub(r"\D", "", value)
                if kind == "iban" and not iban_ok(value):
                    continue
                if kind == "credit_card" and not (13 <= len(digits) <= 19 and luhn_ok(digits)):
                    continue
                if kind == "codice_fiscale" and not codice_fiscale_ok(value):
                    continue
                if kind == "phone" and not 8 <= len(digits) <= 15:
                    continue
                found.append(Finding(category, kind, start, end))
                taken.append((start, end))
        return sorted(found, key=lambda f: f.start)


@dataclass
class PIIResult:
    action: str  # allow | redact | block | reject
    text: str  # the text to store (redacted when action == "redact")
    findings: list[Finding] = field(default_factory=list)

    @property
    def stored(self) -> bool:
        return self.action in ("allow", "redact")

    def reason(self) -> str:
        kinds = sorted({f.kind for f in self.findings})
        return f"{self.action}: {', '.join(kinds)}" if kinds else self.action


class PIIFilter:
    """Applies the detectors and the per-level policy. `shared` = a team or org scope (or a promotion)."""

    def __init__(self, detectors: list | None = None, user_personal: str = "allow", shared_personal: str = "redact"):
        self.detectors = list(detectors) if detectors is not None else [RegexDetector()]
        self.user_personal = user_personal
        self.shared_personal = shared_personal

    @classmethod
    def from_config(cls, pii_config, detectors: list | None = None) -> PIIFilter:
        return cls(detectors, pii_config.user_scope_personal, pii_config.shared_scope_personal)

    def findings(self, text: str) -> list[Finding]:
        out: list[Finding] = []
        for detector in self.detectors:
            for f in detector.find(text):
                if not any(o.start < f.end and f.start < o.end for o in out):
                    out.append(f)
        return sorted(out, key=lambda f: f.start)

    def check(self, text: str, shared: bool) -> PIIResult:
        findings = self.findings(text)
        if not findings:
            return PIIResult("allow", text)
        if any(f.category == SECRET for f in findings):
            return PIIResult("reject", "", findings)
        policy = self.shared_personal if shared else self.user_personal
        if policy == "allow":
            return PIIResult("allow", text, findings)
        if policy == "block":
            return PIIResult("block", text, findings)
        redacted = text
        for f in sorted(findings, key=lambda f: f.start, reverse=True):
            redacted = redacted[: f.start] + f"[{f.kind.upper()}]" + redacted[f.end :]
        return PIIResult("redact", redacted, findings)
