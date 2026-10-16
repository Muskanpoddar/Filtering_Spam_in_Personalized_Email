"""Interactive demo: load trained personalized detector and classify new emails with feedback loop."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import parse_email_from_path  # noqa: E402
from src.features.structural import extract_batch  # noqa: E402
from src.models.personalized import PersonalizedSpamDetector  # noqa: E402
from src.preprocessor import ParsedEmail  # noqa: E402


SAMPLE_EMAILS: list[dict] = [
    {
        "path": None,
        "subject": "WIN A FREE iPHONE 15 PRO!!!",
        "sender": "lottery@cheap-deals.click",
        "body": ("Congratulations!!! You have been selected as the lucky winner of an iPhone 15 Pro. "
                 "Click here http://cheap-deals.click/claim?id=42 to claim your prize within 24 hours. "
                 "FREE!! Limited offer. http://bit.ly/free-iphone"),
    },
    {
        "path": None,
        "subject": "Quarterly review meeting",
        "sender": "manager@company.com",
        "body": ("Hi team, just a reminder that our quarterly review meeting is scheduled for Friday at 3 PM. "
                 "Please bring your status updates and any blockers. Let me know if the time doesn't work."),
    },
    {
        "path": None,
        "subject": "Re: budget approval",
        "sender": "finance@partner-firm.com",
        "body": ("Attached please find the revised budget for the partnership program. "
                 "We still need sign-off from the VP before we can proceed. Let me know if you have any questions."),
    },
    {
        "path": None,
        "subject": "URGENT!! Verify your account to avoid suspension",
        "sender": "security-no-reply@bankk-secure-login.tk",
        "body": ("Dear user your account will be deactivated. Please verify your details immediately: "
                 "http://bankk-secure-login.tk/login?id=xyz"),
    },
    {
        "path": None,
        "subject": "Lunch on Wednesday?",
        "sender": "alex@friend.com",
        "body": ("Hey! Are you free for lunch on Wednesday? Wanted to try the new Thai place near the office."),
    },
]


def _emails_from_args(paths: Iterable[str]) -> list[ParsedEmail]:
    emails: list[ParsedEmail] = []
    for p in paths:
        parsed = parse_email_from_path(Path(p))
        parsed.sender = parsed.sender or ""
        parsed.subject = parsed.subject or ""
        parsed.body = parsed.body or parsed.raw_body
        emails.append(parsed)
    return emails


def _sample_emails() -> list[ParsedEmail]:
    out: list[ParsedEmail] = []
    for s in SAMPLE_EMAILS:
        e = ParsedEmail()
        e.subject = s["subject"]
        e.sender = s["sender"]
        e.body = s["body"]
        e.raw_body = s["body"]
        e.word_count = len(s["body"].split())
        e.char_count = len(s["body"])
        e.urls = []
        out.append(e)
    return out


def _format(e: ParsedEmail, score: float, prediction: int) -> str:
    label = "SPAM" if prediction == 1 else "HAM"
    return (f"[{label:>4}] score={score:.3f} | from={e.sender or '<unknown>':<35s} "
            f"| subject={(e.subject or '')[:55]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models/personalized"))
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--use-samples", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading personalized detector from {args.models_dir}...")
    detector = PersonalizedSpamDetector.load(str(args.models_dir))
    print("Profile:", detector.profile.summary())

    if args.files:
        emails = _emails_from_args(args.files)
    else:
        emails = _sample_emails()
    struct = extract_batch(emails)
    probs = detector.predict_proba(emails, struct)
    preds = (probs >= detector.threshold).astype(int)

    for e, p, y in zip(emails, probs, preds):
        print(_format(e, float(p), int(y)))

    print("\nProvide feedback ('y' = spam, 'n' = ham, 's' = skip, 'q' = quit):")
    for e, p, y in zip(emails, probs, preds):
        ans = input(f"  Was '{e.subject[:40]}' spam? [y/n/s/q] (pred={y}, score={float(p):.2f}): ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        if ans in {"y", "yes"}:
            detector.provide_feedback(e, 1, note="user_marked_spam")
        elif ans in {"n", "no"}:
            detector.provide_feedback(e, 0, note="user_marked_ham")
        else:
            continue
    detector.save(str(args.models_dir))
    print(f"\nProfile updated and saved to {args.models_dir}/profile.json")


if __name__ == "__main__":
    main()