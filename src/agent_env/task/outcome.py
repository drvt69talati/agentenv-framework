"""What a step failure means for its run: the role the failed step played, why it failed, and whether the completed
run is degraded.

A tolerated failure (``fail_task_on_error=False``) lets the run complete, so these facts are the only record of what
the run lost. Each ``context.metadata.failed_steps`` entry carries them, and a completed run's ``quality`` sums them up.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from agent_env.task_step.task_step import StepRole

if TYPE_CHECKING:
    from agent_env.task_step.task_step import TaskStep

__all__ = ["FailedStep", "FailureCause", "RunQuality", "StepRole", "classify_failure", "run_quality"]


class FailureCause(StrEnum):
    KEY_BUDGET = "key_budget"
    KEY_AUTH = "key_auth"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


# Checked in order: a key that is out of budget or rejected fails every retry, so it wins over a timeout that wraps it.
_CAUSE_PATTERNS: tuple[tuple[FailureCause, re.Pattern[str]], ...] = (
    (FailureCause.KEY_BUDGET, re.compile(
        r"budget\s+(?:has\s+been\s+)?exceeded|exceeded\s+(?:the\s+|your\s+)?(?:max\s+)?budget|ExceededBudget|BudgetExceeded",
        re.IGNORECASE,
    )),
    (FailureCause.KEY_AUTH, re.compile(
        r"\b401\b|unauthori[sz]ed|authentication\s*error|token_not_found_in_db|incorrect\s+api\s+key"
        r"|invalid[\s_-]+(?:proxy[\s_-]+server[\s_-]+)?(?:api[\s_-]+)?(?:key|token)"
        r"|(?:key|token)\s+(?:has\s+)?(?:been\s+)?(?:revoked|expired)|(?:revoked|expired)\s+(?:api[\s_-]+)?(?:key|token)",
        re.IGNORECASE,
    )),
    (FailureCause.TRANSIENT, re.compile(
        r"timed?[\s_-]*out|disconnect|connection\s*(?:reset|refused|aborted|closed|error)|broken\s*pipe|remote\s+end\s+closed"
        r"|internal\s+server\s+error|bad\s+gateway|service\s+unavailable|gateway\s+time-?out"
        r"|(?:HTTP|status(?:[\s_]code)?|error)[\s:=]*5\d\d\b"
        r"|\b429\b|rate[\s_-]?limit|too\s+many\s+requests"
        r"|OOM[\s_-]?kill|out\s+of\s+memory|\bOOM\b|exit(?:ed)?(?:\s+with)?\s+code\s+137",
        re.IGNORECASE,
    )),
)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and every exception it was raised from or during, each once."""
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend([current.__cause__, current.__context__])
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)


def classify_failure(exc: BaseException) -> FailureCause:
    """Why ``exc`` happened, read from the type names and messages of its whole chain.

    Matching text is the only option: a judge or agent error crosses A2A and the model proxy as a message string, so its
    original type is gone by the time a step raises.
    """
    text = "\n".join(f"{type(e).__name__}: {e}" for e in _chain(exc))
    for cause, pattern in _CAUSE_PATTERNS:
        if pattern.search(text):
            return cause
    return FailureCause.UNKNOWN


@dataclass(frozen=True)
class FailedStep:
    """One ``context.metadata.failed_steps`` entry."""

    step_id: str
    step_type: str
    error: str
    error_type: str
    started_at_utc: str
    duration_seconds: float
    is_fatal: bool
    retried: bool = False
    step_role: StepRole = StepRole.OTHER
    cause: FailureCause = FailureCause.UNKNOWN
    # Set once a later action (e.g. a regrade) repaired what this failure lost.
    resolved_by: str | None = None

    @classmethod
    def from_exception(
        cls, step: TaskStep, exc: BaseException, *, started_at_utc: str, duration_seconds: float, is_fatal: bool,
    ) -> FailedStep:
        return cls(
            step_id=step.id,
            step_type=step.type,
            error=str(exc),
            error_type=type(exc).__name__,
            started_at_utc=started_at_utc,
            duration_seconds=duration_seconds,
            is_fatal=is_fatal,
            step_role=getattr(step, "step_role", StepRole.OTHER),
            cause=classify_failure(exc),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FailedStep:
        """An entry as stored. One written before ``step_role`` and ``cause`` existed, or holding a value this version
        does not know, reads as ``other`` and ``unknown``."""
        return cls(
            step_id=data.get("step_id", ""),
            step_type=data.get("step_type", ""),
            error=data.get("error", ""),
            error_type=data.get("error_type", ""),
            started_at_utc=data.get("started_at_utc", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            is_fatal=bool(data.get("is_fatal", True)),
            retried=bool(data.get("retried", False)),
            step_role=StepRole(data["step_role"]) if data.get("step_role") in _ROLES else StepRole.OTHER,
            cause=FailureCause(data["cause"]) if data.get("cause") in _CAUSES else FailureCause.UNKNOWN,
            resolved_by=data.get("resolved_by"),
        )

    def to_dict(self) -> dict[str, Any]:
        """The stored form: the keys every reader already knows, plus ``step_role`` and ``cause``. ``retried`` and
        ``resolved_by`` appear only once set, as ``retried`` always has."""
        out: dict[str, Any] = {
            "step_id": self.step_id, "step_type": self.step_type,
            "error": self.error, "error_type": self.error_type,
            "started_at_utc": self.started_at_utc, "duration_seconds": self.duration_seconds,
            "is_fatal": self.is_fatal,
            "step_role": self.step_role.value, "cause": self.cause.value,
        }
        if self.retried:
            out["retried"] = True
        if self.resolved_by is not None:
            out["resolved_by"] = self.resolved_by
        return out


_ROLES = frozenset(StepRole)
_CAUSES = frozenset(FailureCause)
# Most severe first: a run with no grade is worse than one missing its agent's turn, which is worse than one missing
# evidence.
_SEVERITY = (StepRole.GRADER, StepRole.AGENT, StepRole.EVIDENCE, StepRole.OTHER)


@dataclass(frozen=True)
class RunQuality:
    """How much of a completed run is usable. ``reasons`` holds the roles of the steps whose failure the run tolerated
    and nothing has repaired, most severe first; a run with none is clean."""

    reasons: tuple[StepRole, ...] = ()
    # Derived, never passed, so it cannot disagree with ``reasons``; a field so ``asdict`` matches the stored form.
    degraded: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "degraded", bool(self.reasons))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RunQuality:
        """``quality`` as stored. A run completed before ``quality`` existed has none and reads as clean."""
        roles = {StepRole(r) for r in (data or {}).get("reasons", ()) if r in _ROLES}
        return cls(tuple(r for r in _SEVERITY if r in roles))

    def to_dict(self) -> dict[str, Any]:
        return {"degraded": self.degraded, "reasons": [r.value for r in self.reasons]}


def run_quality(failed_steps: Iterable[FailedStep]) -> RunQuality:
    """A fatal failure fails the run instead, and a retried or resolved one lost nothing."""
    open_roles = {f.step_role for f in failed_steps if not f.is_fatal and not f.retried and f.resolved_by is None}
    return RunQuality(tuple(r for r in _SEVERITY if r in open_roles))
