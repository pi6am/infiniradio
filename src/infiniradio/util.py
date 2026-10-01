"""Small shared helpers: time, logging, math, formatting, locking."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

# All timestamps in the database are naive UTC ISO-8601 strings with fixed
# precision, which makes them lexicographically sortable.
_TS_SPEC = "seconds"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt: datetime) -> str:
    return dt.replace(tzinfo=None).isoformat(timespec=_TS_SPEC)


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def iso_delta(seconds: float) -> str:
    return iso(utcnow() - timedelta(seconds=seconds))


def epoch_ms() -> int:
    """True epoch milliseconds (timezone-safe; utcnow() is naive)."""
    return round(datetime.now(timezone.utc).timestamp() * 1000)


def wilson_score(ups: int, downs: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for the upvote ratio.

    Conservative with few votes: 1 up / 0 down scores well below 1.0.
    Returns 0.0 when there are no votes.
    """
    n = ups + downs
    if n <= 0:
        return 0.0
    p = ups / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(max(0.0, (p * (1 - p) + z2 / (4 * n)) / n))
    return (centre - margin) / denom


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def normalize_newlines(text: str) -> str:
    """CRLF (and lone CR) -> LF, at every text boundary.

    Two proven sources of stray CR: browsers submit <textarea> values with
    CRLF on every platform (the HTML spec's submission normalization — not
    a Windows paste, a browser one), and some LLM providers emit CRLF in
    completions. YuE2's conditioning derailed on CR-mangled lyrics — the
    same lyrics, style and seed rendered correctly with LF and scrambled
    with CRLF (matched pair, ComfyUI history) — so all submitted and
    received text is normalized to Unix newlines before it is stored or
    sent on.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def lock_data_dir(data_dir: Path) -> None:
    """Take an exclusive, process-lifetime lock on the data directory.

    Two servers sharing one data dir fight over the SQLite DB, the HLS
    playlist and the feeder timeline in confusing ways; refuse to start
    instead. Best effort on platforms without a usable lock API.
    """
    import os
    import sys

    lock_path = data_dir / "infiniradio.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        logging.getLogger(__name__).warning("cannot create lock file %s: %s", lock_path, exc)
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, OSError) as exc:
        raise RuntimeError(
            f"another infiniradio instance is already using {data_dir} "
            "(delete infiniradio.lock only if you are certain none is running)"
        ) from exc
    # Deliberately keep fd open (and the module-level reference alive) for
    # the lifetime of the process: the lock releases when we exit.
    globals()["_lock_fd"] = fd
