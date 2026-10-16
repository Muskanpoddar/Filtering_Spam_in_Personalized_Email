"""Email parsing and text cleaning utilities."""
from __future__ import annotations

import email
import email.policy
import re
from dataclasses import dataclass, field
from email.message import Message
from html import unescape
from typing import Optional

from bs4 import BeautifulSoup

URL_RE = re.compile(r"https?://[^\s<>\"']+|ftp://[^\s<>\"']+", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
TOKEN_RE = re.compile(r"[A-Za-z]+")


@dataclass
class ParsedEmail:
    subject: str = ""
    body: str = ""
    sender: str = ""
    sender_domain: str = ""
    reply_to: str = ""
    recipients: list[str] = field(default_factory=list)
    raw_headers: str = ""
    raw_body: str = ""
    urls: list[str] = field(default_factory=list)
    has_html: bool = False
    html_ratio: float = 0.0
    has_attachment: bool = False
    attachment_count: int = 0
    char_count: int = 0
    word_count: int = 0
    num_exclamation: int = 0
    num_question: int = 0
    num_caps_words: int = 0
    num_dollar_signs: int = 0
    num_digits: int = 0
    suspicious_subject: bool = False


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return part.get_payload() or ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="ignore")
    except (LookupError, TypeError):
        return payload.decode("utf-8", errors="ignore")


def _strip_html(text: str) -> tuple[str, float]:
    if not text:
        return "", False
    has_html = bool(HTML_TAG_RE.search(text))
    if has_html:
        soup = BeautifulSoup(text, "lxml")
        plain = soup.get_text(separator=" ")
        ratio = min(1.0, len(text) / max(1, len(plain) + 1))
        return plain, ratio
    return text, 0.0


def _clean_text(text: str) -> str:
    text = unescape(text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip().lower()


def _extract_domain(addr: str) -> str:
    if not addr or "@" not in addr:
        return ""
    return addr.rsplit("@", 1)[-1].strip("> ").lower()


def parse_email(raw_bytes: bytes) -> ParsedEmail:
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.compat32)
    parsed = ParsedEmail()
    parsed.subject = str(msg.get("Subject", "") or "")
    parsed.sender = str(msg.get("From", "") or "")
    parsed.reply_to = str(msg.get("Reply-To", "") or "")
    parsed.sender_domain = _extract_domain(parsed.sender)
    parsed.raw_headers = str(msg)

    tos = [str(t).strip() for t in msg.get_all("To", []) or []]
    ccs = [str(c).strip() for c in msg.get_all("Cc", []) or []]
    parsed.recipients = tos + ccs

    body_chunks: list[str] = []
    html_chunks: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            filename = part.get_filename()
            if filename or "attachment" in disp:
                parsed.has_attachment = True
                parsed.attachment_count += 1
                continue
            if ctype == "text/plain":
                body_chunks.append(_decode_part(part))
            elif ctype == "text/html":
                html_chunks.append(_decode_part(part))
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            body_chunks.append(_decode_part(msg))
        elif ctype == "text/html":
            html_chunks.append(_decode_part(msg))

    text_body = "\n".join(body_chunks)
    html_body = "\n".join(html_chunks)
    if html_body and not text_body:
        parsed.has_html = True
        text_body, ratio = _strip_html(html_body)
        parsed.html_ratio = ratio
    elif html_body and text_body:
        parsed.has_html = True
        plain_html, ratio = _strip_html(html_body)
        parsed.html_ratio = ratio
        text_body = text_body + "\n" + plain_html

    parsed.raw_body = text_body
    parsed.urls = list(dict.fromkeys(URL_RE.findall(parsed.raw_body or "")))
    parsed.body = _clean_text(parsed.raw_body)
    parsed.char_count = len(parsed.raw_body)
    parsed.word_count = len(parsed.body.split()) if parsed.body else 0
    parsed.num_exclamation = (parsed.raw_body or "").count("!")
    parsed.num_question = (parsed.raw_body or "").count("?")
    parsed.num_dollar_signs = (parsed.raw_body or "").count("$")
    parsed.num_digits = sum(ch.isdigit() for ch in parsed.raw_body)
    caps_words = TOKEN_RE.findall(parsed.raw_body or "")
    parsed.num_caps_words = sum(1 for w in caps_words if len(w) > 2 and w.isupper())
    subj = parsed.subject or ""
    parsed.suspicious_subject = bool(
        re.search(r"free|winner|viagra|$$$|% off|urgent|click", subj, re.I)
    )
    return parsed


def parse_email_from_file(path: str) -> ParsedEmail:
    with open(path, "rb") as fh:
        return parse_email(fh.read())


def short_subject_for_features(subject: str) -> str:
    return _clean_text(subject)[:120]