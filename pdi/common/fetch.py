"""Collect step: fetch a response and save it untouched with retrieval time, URL and checksum."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

USER_AGENT = "public-data-intelligence (+https://github.com/liumOazed/public-data-intelligence)"
RETRYABLE = (429, 500, 502, 503, 504)
_SECRET_PARAM = re.compile(r"(?i)\b(api_?key|key|token)=[^&\s]+")


class CollectError(RuntimeError):
    """A collector could not get a usable response. The run must stop."""


def redact(url: str) -> str:
    """Strip API keys from a URL before it is logged or written to disk."""
    return _SECRET_PARAM.sub(lambda m: f"{m.group(1)}=REDACTED", url)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def fetch(
    url: str,
    dest: Path,
    *,
    params: dict | None = None,
    retries: int = 4,
    timeout: int = 120,
    session: requests.Session | None = None,
) -> Path:
    """Download `url` to `dest` byte for byte and write `dest`.meta.json beside it."""
    http = session or requests.Session()
    resp = None
    last = "no attempt made"
    for attempt in range(retries):
        if attempt:
            time.sleep(2**attempt)
        try:
            resp = http.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            last = type(exc).__name__
            continue
        if resp.status_code == 200:
            break
        last = f"HTTP {resp.status_code}: {resp.text[:200]!r}"
        if resp.status_code not in RETRYABLE:
            # Other client errors (bad key, bad query) do not get better on retry.
            raise CollectError(f"{redact(resp.url)} answered {last}")
    else:
        raise CollectError(f"{redact(url)} failed after {retries} attempts: {last}")

    body = resp.content
    if not body:
        raise CollectError(f"{redact(resp.url)} returned an empty body")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    meta = {
        "url": redact(resp.url),
        "retrieved_at": utc_now().isoformat(),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
        "content_type": resp.headers.get("Content-Type"),
    }
    Path(str(dest) + ".meta.json").write_text(json.dumps(meta, indent=2))
    return dest


def read_meta(path: Path) -> dict:
    return json.loads(Path(str(path) + ".meta.json").read_text())


def latest(directory: Path, pattern: str) -> Path:
    """Newest raw snapshot matching `pattern`, by name (names start with the retrieval date)."""
    files = sorted(p for p in directory.glob(pattern) if not p.name.endswith(".meta.json"))
    if not files:
        raise CollectError(f"no raw snapshot matching {pattern} in {directory}")
    return files[-1]
