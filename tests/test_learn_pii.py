"""pgs learn PII filter: regex baseline with checksums, per-scope policy, promotion re-check."""

import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

from palingenesis.learn import LearnConfig, Learner  # noqa: E402
from palingenesis.learn.pii import PIIFilter, RegexDetector, codice_fiscale_ok, iban_ok, luhn_ok  # noqa: E402

USER = "user:acme/support/ana"


def kinds(text):
    return [f.kind for f in RegexDetector().find(text)]


@pytest.mark.parametrize(
    "text, kind",
    [
        ("write to ana.rossi@example.com today", "email"),
        ("call me at +39 347 123 4567", "phone"),
        ("my mobile is 347 1234567", "phone"),
        ("the office is 02 12345678", "phone"),
        ("US office (555) 123-4567", "phone"),
        ("codice fiscale RSSMRA85T10A562S", "codice_fiscale"),
        ("IBAN IT60 X054 2811 1010 0000 0123 456", "iban"),
        ("IBAN IT60X0542811101000000123456", "iban"),
        ("card 4111 1111 1111 1111", "credit_card"),
        ("card 4111-1111-1111-1111 exp 12/29", "credit_card"),
        ("token sk-abcdefghijklmnopqrstuvwxyz123456", "api_key"),
        ("key AKIAABCDEFGHIJKLMNOP", "api_key"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private_key"),
        ("password: hunter22", "password"),
    ],
)
def test_detects(text, kind):
    assert kind in kinds(text)


@pytest.mark.parametrize(
    "text",
    [
        "The price of Orion Pro is 25 euros.",
        "Released on 2024-01-15, version 0.26.0.",
        "Order 4111 1111 1111 1112 failed",  # fails Luhn
        "IT60X0542811101000000123457",  # fails mod 97
        "RSSMRA85T10A562X",  # wrong control character
        "The limit is 1000000 tokens and 128 GB.",
        "Port 8000, max 64 LoRAs, 3.14159",
    ],
)
def test_ignores(text):
    assert kinds(text) == []


def test_checksums():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82 WEST 1234 5698 7654 32") and not iban_ok("GB82 WEST 1234 5698 7654 33")
    assert codice_fiscale_ok("RSSMRA85T10A562S") and not codice_fiscale_ok("RSSMRA85T10A562T")


def test_policy_per_scope_level():
    pii = PIIFilter(user_personal="allow", shared_personal="redact")
    text = "Ana's email is ana@example.com"
    assert pii.check(text, shared=False).action == "allow"
    shared = pii.check(text, shared=True)
    assert shared.action == "redact" and shared.text == "Ana's email is [EMAIL]"
    assert pii.check("card 4111 1111 1111 1111", shared=False).action == "reject"
    assert PIIFilter(shared_personal="block").check(text, shared=True).action == "block"
    assert pii.check("nothing here", shared=True).action == "allow"


def test_pluggable_detector():
    from palingenesis.learn.pii import PERSONAL, Finding

    class Names:
        def find(self, text):
            i = text.find("Mario")
            return [Finding(PERSONAL, "name", i, i + 5)] if i >= 0 else []

    pii = PIIFilter([RegexDetector(), Names()], shared_personal="redact")
    assert pii.check("Mario wrote ana@example.com", shared=True).text == "[NAME] wrote [EMAIL]"


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


def store_bytes(learner):
    return b"".join(p.read_bytes() for p in learner.store.root.rglob("*") if p.is_file())


def test_secrets_never_stored(learner, tmp_path):
    result = learner.remember(USER, "my card is 4111 1111 1111 1111")
    assert result.status == "rejected"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(
        "# Ops\nThe admin password: hunter22x\n\n# Pricing\nThe price of Orion Pro is 25 euros.\n"
    )
    report = learner.add_documents("org:acme", "*.md", root=tmp_path / "docs", approve=True)
    assert report.new == 1 and len(report.rejected) == 1
    data = store_bytes(learner)
    assert b"4111" not in data and b"hunter22x" not in data


def test_personal_data_redacted_at_shared_scopes_before_storage(learner, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Team\nThe contact of Orion support is ana@example.com.\n")
    learner.add_documents("team:acme/support", "*.md", root=tmp_path / "docs")  # waits in the inbox
    assert b"ana@example.com" not in store_bytes(learner)
    learner.remember(USER, "my email is ana@example.com")
    assert learner.store.scope(USER).active_notes()[0].text == "my email is ana@example.com"


def test_promotion_rechecks_pii_at_the_shared_level(learner):
    stored = learner.remember(USER, "The Orion escalation contact is ana@example.com")
    queue_id = learner.promote(USER, stored.note_id, "org:acme")
    learner.approve("org:acme", [queue_id])
    assert learner.store.scope("org:acme").active_notes()[0].text == "The Orion escalation contact is [EMAIL]"
