"""Admin authentication: scrypt passwords, signed session cookies, CSRF.

Single admin account: username from config, password hash in the DB settings
table (set via `python -m infiniradio passwd`) falling back to a hash in the
config file. Sessions are signed cookies (itsdangerous, 7-day max age).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..config import Config
from ..db import DB

SESSION_COOKIE = "ir_admin"
SESSION_MAX_AGE = 7 * 24 * 3600
MAX_ATTEMPTS = 5
LOCKOUT_S = 300

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def get_server_secret(db: DB, cfg: Config) -> str:
    secret = db.get_setting("server.secret") or cfg.get("server.secret", "")
    if not secret:
        secret = secrets.token_hex(32)
        db.set_setting("server.secret", secret)
    return str(secret)


def stored_password_hash(db: DB, cfg: Config) -> str:
    """DB setting wins over config file."""
    return str(db.get_setting("admin.password_hash") or cfg.get("admin.password_hash", ""))


class Sessions:
    def __init__(self, secret: str):
        self._signer = URLSafeTimedSerializer(secret, salt="ir-admin-session")
        self._csrf_key = (secret + "|csrf").encode()

    def issue(self) -> str:
        return self._signer.dumps({"sid": secrets.token_hex(16), "ts": time.time()})

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        try:
            self._signer.loads(token, max_age=SESSION_MAX_AGE)
            return True
        except (BadSignature, SignatureExpired):
            return False

    def csrf_token(self, token: str | None) -> str:
        """CSRF token bound to the session cookie value."""
        return hmac.new(self._csrf_key, (token or "").encode(), hashlib.sha256).hexdigest()

    def check_csrf(self, token: str | None, submitted: str | None) -> bool:
        return hmac.compare_digest(self.csrf_token(token), submitted or "")


class LoginThrottle:
    def __init__(self) -> None:
        self._attempts: dict[str, tuple[int, float]] = {}

    def locked(self, key: str) -> bool:
        entry = self._attempts.get(key)
        if not entry:
            return False
        fails, until = entry
        if fails >= MAX_ATTEMPTS and time.time() < until:
            return True
        return False

    def record_failure(self, key: str) -> None:
        fails, _ = self._attempts.get(key, (0, 0.0))
        self._attempts[key] = (fails + 1, time.time() + LOCKOUT_S)

    def reset(self, key: str) -> None:
        self._attempts.pop(key, None)

    def seconds_left(self, key: str) -> int:
        entry = self._attempts.get(key)
        if not entry:
            return 0
        fails, until = entry
        return max(0, int(until - time.time())) if fails >= MAX_ATTEMPTS else 0
