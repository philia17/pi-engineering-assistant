"""Opt-in access-key login and in-memory session layer for the Engineering Preview.

FEI3-02: when ``EEE_WEB_DEMO_AUTH_REQUIRED=1``, a device presents one invite
(access) key at ``POST /auth/session`` and receives a short-lived HttpOnly
session cookie. The key itself is never stored, logged, or echoed: verification
is a constant-time comparison against a repository-external PBKDF2-HMAC-SHA256
digest configured through environment variables.

This is a preview-only, memory-only session store. It is not production
persistence, multi-user identity management, or a production anti-abuse
layer; FEI3-08 adds a small in-memory failed-login rate limit so the stable
web demo cannot be brute-forced during the public preview.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from _thread import LockType
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from web_ui.quiz_sessions import QuizState

SESSION_COOKIE_NAME: str = "eee_demo_session"
AUTH_REQUIRED_ENV: str = "EEE_WEB_DEMO_AUTH_REQUIRED"
ACCESS_SALT_ENV: str = "EEE_WEB_DEMO_ACCESS_SALT"
ACCESS_DIGEST_ENV: str = "EEE_WEB_DEMO_ACCESS_DIGEST"
SESSION_TTL_SECONDS_ENV: str = "EEE_WEB_DEMO_SESSION_TTL_SECONDS"
COOKIE_SECURE_ENV: str = "EEE_WEB_DEMO_COOKIE_SECURE"
DEFAULT_SESSION_TTL_SECONDS: int = 30 * 60
PBKDF2_HASH_NAME: str = "sha256"
PBKDF2_ITERATIONS: int = 600_000
MIN_SALT_BYTES: int = 16
DIGEST_BYTES: int = 32
RATE_LIMIT_WINDOW_SECONDS: int = 60
RATE_LIMIT_MAX_FAILED_ATTEMPTS: int = 5
RATE_LIMIT_MAX_CLIENTS: int = 10_000
RATE_LIMIT_FALLBACK_CLIENT_HOST: str = "unknown"
RATE_LIMIT_DETAIL_CODE: str = "too_many_attempts"
RATE_LIMIT_DETAIL_MESSAGE: str = "Too many failed sign-in attempts. Try again later."
MAX_CONVERSATION_TURNS: int = 6
MAX_CONVERSATION_CONTEXT_CHARS: int = 6_000
CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
AUTH_COOKIE_NOT_SECURE_CODE: str = "auth_cookie_not_secure"
AUTH_COOKIE_NOT_SECURE_MESSAGE: str = (
    "Demo auth requires HTTPS Secure cookies (EEE_WEB_DEMO_COOKIE_SECURE=1)."
)

_sessions: dict[str, DemoSession] = {}
_failed_attempts: dict[str, list[float]] = {}


@dataclass(frozen=True)
class QuizStartReservation:
    """One session-owned reservation for the first Quiz question."""

    token: str
    generation: int
    previous_quiz_state: QuizState | None


@dataclass
class DemoSession:
    """Server-held demo session record (memory-only, not production persistence)."""

    session_id: str
    created_at: float
    expires_at: float
    conversation_id: str | None = None
    conversation_history: list[tuple[str, str]] | None = None
    quiz_state: QuizState | None = None
    quiz_fact_history: dict[tuple[str, str, str, str | None], tuple[str, ...]] = field(
        default_factory=dict
    )
    quiz_start_in_flight: str | None = None
    quiz_start_generation: int = 0
    invalidated: bool = False
    _quiz_start_lock: LockType = field(default_factory=Lock, repr=False)

    def reserve_quiz_start(self) -> QuizStartReservation | None:
        """Atomically reserve this session's first-question generation."""
        with self._quiz_start_lock:
            if (
                self.invalidated
                or self.expires_at <= time.time()
                or self.quiz_start_in_flight is not None
                or (self.quiz_state is not None and self.quiz_state.status == "in_progress")
            ):
                return None
            reservation = QuizStartReservation(
                token=secrets.token_urlsafe(18),
                generation=self.quiz_start_generation,
                previous_quiz_state=self.quiz_state,
            )
            self.quiz_start_in_flight = reservation.token
            return reservation

    def install_quiz_start(self, reservation: QuizStartReservation, quiz: QuizState) -> bool:
        """Install a generated Quiz only if its session reservation is still live."""
        with self._quiz_start_lock:
            valid = (
                not self.invalidated
                and self.expires_at > time.time()
                and self.quiz_start_generation == reservation.generation
                and self.quiz_start_in_flight == reservation.token
                and self.quiz_state is reservation.previous_quiz_state
            )
            if valid:
                self.quiz_state = quiz
                self.quiz_fact_history[quiz.scope_key] = (quiz.current.question.fact_id,)
            if self.quiz_start_in_flight == reservation.token:
                self.quiz_start_in_flight = None
            return valid

    def previous_quiz_fact_ids(
        self, scope_key: tuple[str, str, str, str | None]
    ) -> tuple[str, ...]:
        """Return the immediately previous attempt's facts for one exact scope."""
        with self._quiz_start_lock:
            return self.quiz_fact_history.get(scope_key, ())

    def record_quiz_fact(self, quiz: QuizState) -> None:
        """Append one installed question to the current scope's freshness history."""
        with self._quiz_start_lock:
            if self.quiz_state is quiz:
                self.quiz_fact_history[quiz.scope_key] = tuple(quiz.generated_fact_ids)

    def release_quiz_start(self, reservation: QuizStartReservation) -> None:
        """Release a first-question reservation after generation finishes."""
        with self._quiz_start_lock:
            if self.quiz_start_in_flight == reservation.token:
                self.quiz_start_in_flight = None

    def invalidate_quiz(self) -> None:
        """Invalidate an in-flight Quiz start and clear the current Quiz."""
        with self._quiz_start_lock:
            self.quiz_start_generation += 1
            self.quiz_start_in_flight = None
            self.quiz_state = None

    def invalidate_session(self) -> None:
        """Invalidate this session and any Quiz start still generating."""
        with self._quiz_start_lock:
            self.invalidated = True
            self.quiz_start_generation += 1
            self.quiz_start_in_flight = None
            self.quiz_state = None
            self.quiz_fact_history.clear()


@dataclass(frozen=True)
class _CredentialConfig:
    """Validated repository-external salt and PBKDF2 digest used for verification."""

    salt: bytes
    digest: bytes


def _env_flag(name: str) -> bool:
    """Return whether an environment variable carries an explicit enable flag."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def auth_required_enabled() -> bool:
    """Return whether the opt-in demo auth gate is enabled."""
    return _env_flag(AUTH_REQUIRED_ENV)


def cookie_secure_enabled() -> bool:
    """Return whether the session cookie should carry the Secure attribute.

    Defaults to False so local HTTP smoke tests keep working; HTTPS staging must
    set ``EEE_WEB_DEMO_COOKIE_SECURE=1``.
    """
    return _env_flag(COOKIE_SECURE_ENV)


def auth_cookie_preflight_error() -> HTTPException | None:
    """Return the fail-closed 503 when auth is enabled without Secure cookies.

    Returns None when auth is disabled (the legacy open surface is unchanged)
    or when ``EEE_WEB_DEMO_COOKIE_SECURE`` is an explicit truthy value. When
    auth is required on insecure staging, the returned error is raised before
    any access-key verification, session issuance, or protected-route unlock.
    The detail carries only the stable non-secret code and message, never
    key/salt/digest material.
    """
    if not auth_required_enabled() or cookie_secure_enabled():
        return None
    return HTTPException(
        status_code=503,
        detail={
            "code": AUTH_COOKIE_NOT_SECURE_CODE,
            "message": AUTH_COOKIE_NOT_SECURE_MESSAGE,
        },
    )


def session_ttl_seconds() -> int:
    """Return the configured session lifetime in seconds (bounded, fail-safe)."""
    raw = os.getenv(SESSION_TTL_SECONDS_ENV, "").strip()
    if not raw:
        return DEFAULT_SESSION_TTL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SESSION_TTL_SECONDS
    if value < 60 or value > 24 * 60 * 60:
        return DEFAULT_SESSION_TTL_SECONDS
    return value


def _load_credential_config() -> _CredentialConfig | None:
    """Load the repository-external PBKDF2 config, failing closed on unsafe input.

    Returns None when the salt/digest are absent, malformed, too short, or all
    zero so the server refuses login and every protected route without leaking
    which condition failed.
    """
    raw_salt = os.getenv(ACCESS_SALT_ENV, "").strip()
    raw_digest = os.getenv(ACCESS_DIGEST_ENV, "").strip()
    if not raw_salt or not raw_digest:
        return None
    try:
        salt = bytes.fromhex(raw_salt)
        digest = bytes.fromhex(raw_digest)
    except ValueError:
        return None
    if len(salt) < MIN_SALT_BYTES or len(digest) != DIGEST_BYTES:
        return None
    if not any(salt) or not any(digest):
        return None
    return _CredentialConfig(salt=salt, digest=digest)


def derive_access_digest_hex(access_key: str, salt_hex: str) -> str:
    """Ops helper: print the PBKDF2-HMAC-SHA256 hex digest for an external key.

    Only the returned hex digest (and the salt hex) belong in the
    repository-external environment file; the access key itself must never be
    persisted, logged, or placed in source control.
    """
    salt = bytes.fromhex(salt_hex)
    derived = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_NAME,
        access_key.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return derived.hex()


def _verify_access_key(access_key: str) -> bool:
    """Verify one access key in constant time against the configured digest."""
    config = _load_credential_config()
    if config is None:
        return False
    derived = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_NAME,
        access_key.encode("utf-8"),
        config.salt,
        PBKDF2_ITERATIONS,
    )
    return hmac.compare_digest(derived, config.digest)


def _prune_expired(now: float) -> None:
    """Remove expired records so the in-memory store stays bounded."""
    for session_id in [sid for sid, record in _sessions.items() if record.expires_at <= now]:
        record = _sessions.pop(session_id, None)
        if record is not None:
            record.invalidate_session()


def _create_session(now: float) -> DemoSession:
    """Create and store a fresh session record with a server-side expiry."""
    record = DemoSession(
        session_id=secrets.token_urlsafe(32),
        created_at=now,
        expires_at=now + session_ttl_seconds(),
        conversation_history=[],
    )
    _sessions[record.session_id] = record
    return record


def _lookup_session(session_id: str | None, now: float) -> DemoSession | None:
    """Return a live session or None; expired sessions are removed."""
    if not session_id:
        return None
    record = _sessions.get(session_id)
    if record is None:
        return None
    if record.expires_at <= now:
        _sessions.pop(session_id, None)
        record.invalidate_session()
        return None
    return record


def _destroy_session(session_id: str | None) -> None:
    """Invalidate a session record by id (idempotent)."""
    if session_id:
        record = _sessions.pop(session_id, None)
        if record is not None:
            record.invalidate_session()


def _client_host(request: Request) -> str:
    """Return the direct peer address, falling back to a safe constant.

    The limiter keys on ``request.client.host`` only; no forwarded-header
    trust is added because that header is trivially spoofable on the demo
    origin.
    """
    if request.client is not None and request.client.host:
        return request.client.host
    return RATE_LIMIT_FALLBACK_CLIENT_HOST


def _prune_failed_attempts(now: float) -> None:
    """Drop out-of-window attempts and evict least-recent clients beyond the cap.

    Keeps the in-memory failed-attempt map bounded regardless of how many
    distinct addresses hit the demo. Eviction is deterministic: the client
    with the oldest last attempt is removed first, with ties resolving to the
    client inserted earliest.
    """
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    for client_host, attempts in list(_failed_attempts.items()):
        kept = [attempt_at for attempt_at in attempts if attempt_at > window_start]
        if kept:
            _failed_attempts[client_host] = kept
        else:
            _failed_attempts.pop(client_host, None)
    while len(_failed_attempts) > RATE_LIMIT_MAX_CLIENTS:
        oldest_host = min(_failed_attempts, key=lambda host: _failed_attempts[host][-1])
        _failed_attempts.pop(oldest_host, None)


def _record_failed_attempt(client_host: str, now: float) -> None:
    """Record one failed key attempt for a client within the sliding window."""
    _prune_failed_attempts(now)
    _failed_attempts.setdefault(client_host, []).append(now)
    _prune_failed_attempts(now)


def _is_rate_limited(client_host: str, now: float) -> bool:
    """Return whether a client has exhausted the demo failed-attempt budget."""
    _prune_failed_attempts(now)
    return len(_failed_attempts.get(client_host, ())) >= RATE_LIMIT_MAX_FAILED_ATTEMPTS


def _reset_failed_attempts(client_host: str) -> None:
    """Clear a client's failed-attempt budget after a successful login."""
    _failed_attempts.pop(client_host, None)


def _iso_timestamp(epoch_seconds: float) -> str:
    """Format an epoch timestamp as an ISO-8601 UTC string."""
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _set_session_cookie(response: Response, record: DemoSession, now: float) -> None:
    """Attach the short-lived HttpOnly session cookie to a response."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=record.session_id,
        max_age=max(1, int(record.expires_at - now)),
        expires=datetime.fromtimestamp(record.expires_at, tz=timezone.utc),
        path="/",
        secure=cookie_secure_enabled(),
        httponly=True,
        samesite="lax",
    )


def _clear_session_cookie(response: Response) -> None:
    """Expire the session cookie with the same attributes used to set it."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=cookie_secure_enabled(),
        httponly=True,
        samesite="lax",
    )


async def demo_session(request: Request) -> DemoSession | None:
    """FastAPI dependency: gate a route when demo auth is enabled.

    With auth disabled this is a no-op so the active Pi :8000 behavior is
    unchanged. With auth enabled it fails closed: missing/unsafe credential
    configuration returns 503 and a missing, invalid, or expired cookie returns
    401.
    """
    if not auth_required_enabled():
        return None
    preflight_error = auth_cookie_preflight_error()
    if preflight_error is not None:
        raise preflight_error
    if _load_credential_config() is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "auth_not_configured", "message": "Demo auth is not configured."},
        )
    now = time.time()
    record = _lookup_session(request.cookies.get(SESSION_COOKIE_NAME), now)
    if record is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "not_authenticated", "message": "A valid demo session is required."},
        )
    return record


def bind_or_validate_conversation(
    session: DemoSession,
    conversation_id: str | None,
) -> str | None:
    """Bind the first client conversation id to a session and reject mismatches.

    Conversation ids are never trusted across sessions: each session owns its
    own binding, and a client cannot steer an existing session to another id.
    """
    if conversation_id is None:
        return None
    if CONVERSATION_ID_PATTERN.fullmatch(conversation_id) is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_conversation_id",
                "message": "conversation_id must be a bounded opaque identifier.",
            },
        )
    if session.conversation_id is None:
        session.conversation_id = conversation_id
        return conversation_id
    if session.conversation_id != conversation_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "conversation_mismatch",
                "message": "conversation_id does not match the demo session binding.",
            },
        )
    return conversation_id


def conversation_context(session: DemoSession | None) -> tuple[tuple[str, str], ...]:
    """Return the bounded, in-memory prior-turn context for one session."""
    if session is None or not session.conversation_history:
        return ()
    return tuple(session.conversation_history)


def record_conversation_turn(session: DemoSession | None, question: str, answer: str) -> None:
    """Retain only bounded text history; prompts/answers never reach browser storage or logs."""
    if session is None:
        return
    history = session.conversation_history
    if history is None:
        history = []
        session.conversation_history = history
    history.append((question[:2_000], answer[:2_000]))
    del history[:-MAX_CONVERSATION_TURNS]
    total = 0
    start = 0
    for index in range(len(history) - 1, -1, -1):
        total += len(history[index][0]) + len(history[index][1])
        if total > MAX_CONVERSATION_CONTEXT_CHARS:
            start = index + 1
            break
    del history[:start]


def reset_conversation(session: DemoSession | None) -> None:
    """Clear the server binding and text history for a New chat request."""
    if session is None:
        return
    session.conversation_id = None
    session.conversation_history = []
    session.invalidate_quiz()


def recover_conversation_binding(session: DemoSession | None) -> None:
    """Clear only the current session's conversation binding for mismatch recovery.

    Unlike ``reset_conversation``, the bounded server-side
    ``conversation_history`` is preserved so the recovered thread keeps its
    teaching context. The operation is idempotent: clearing an already
    unbound session leaves the history untouched and remains a single
    effective state transition.
    """
    if session is None:
        return
    session.conversation_id = None


auth_router: APIRouter = APIRouter(prefix="/auth", tags=["demo-auth"])


@auth_router.post("/conversation/reset")
def reset_conversation_route(
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, str]:
    """Clear the current session's conversation binding and bounded history."""
    reset_conversation(session)
    return {"status": "conversation_reset"}


@auth_router.post("/conversation/recover")
def recover_conversation_route(
    session: Annotated[DemoSession | None, Depends(demo_session)] = None,
) -> dict[str, str]:
    """Clear only the current session's binding while preserving its history.

    The session is resolved exclusively from the authenticated cookie through
    ``Depends(demo_session)``; no client-selected session or conversation id
    is accepted. With auth enabled, a missing, invalid, or expired session
    fails closed (401/503) before any mutation; with auth disabled the route
    is a harmless no-op like the legacy open preview surface.
    """
    recover_conversation_binding(session)
    return {"status": "conversation_recovered"}


@auth_router.post("/session")
def create_session(
    access_key: Annotated[str, Form()],
    request: Request,
) -> Response:
    """Exchange one invite/access key for a short-lived HttpOnly session cookie.

    Failed attempts are counted per client address in a bounded sliding
    window. The budget is checked before key verification so the 429 response
    never depends on key validity, and the raw key is never stored or logged.
    """
    if not auth_required_enabled():
        raise HTTPException(
            status_code=404,
            detail={"code": "auth_disabled", "message": "Demo auth is not enabled."},
        )
    preflight_error = auth_cookie_preflight_error()
    if preflight_error is not None:
        raise preflight_error
    if _load_credential_config() is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "auth_not_configured", "message": "Demo auth is not configured."},
        )
    client_host = _client_host(request)
    now = time.time()
    if _is_rate_limited(client_host, now):
        raise HTTPException(
            status_code=429,
            detail={
                "code": RATE_LIMIT_DETAIL_CODE,
                "message": RATE_LIMIT_DETAIL_MESSAGE,
            },
        )
    if not _verify_access_key(access_key):
        _record_failed_attempt(client_host, now)
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_access_key", "message": "Invalid access key."},
        )
    _reset_failed_attempts(client_host)
    _prune_expired(now)
    record = _create_session(now)
    response = JSONResponse(
        {
            "status": "ok",
            "expires_at": _iso_timestamp(record.expires_at),
            "ttl_seconds": session_ttl_seconds(),
        }
    )
    _set_session_cookie(response, record, now)
    return response


@auth_router.post("/logout")
def logout(request: Request) -> Response:
    """Invalidate the current session (if any) and clear the cookie."""
    _destroy_session(request.cookies.get(SESSION_COOKIE_NAME))
    response = JSONResponse({"status": "logged_out"})
    _clear_session_cookie(response)
    return response


@auth_router.get("/me")
def session_me(request: Request) -> dict[str, object]:
    """Report the current demo session when auth is enabled and valid."""
    if not auth_required_enabled():
        raise HTTPException(
            status_code=401,
            detail={"code": "not_authenticated", "message": "Demo auth is not enabled."},
        )
    preflight_error = auth_cookie_preflight_error()
    if preflight_error is not None:
        raise preflight_error
    if _load_credential_config() is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "auth_not_configured", "message": "Demo auth is not configured."},
        )
    now = time.time()
    record = _lookup_session(request.cookies.get(SESSION_COOKIE_NAME), now)
    if record is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "not_authenticated", "message": "A valid demo session is required."},
        )
    return {
        "authenticated": True,
        "expires_at": _iso_timestamp(record.expires_at),
        "ttl_seconds": max(0, int(record.expires_at - now)),
        "conversation_bound": record.conversation_id is not None,
    }
