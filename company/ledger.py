"""Dual budget ledger: portable LLM-token units + Modal GPU credits.

Credits are provider-reported compute costs. Token values are the portable
chars/4 estimates and requested-output reservations produced by the company
hierarchy, not exact provider billing tokens: the common backend interface does
not expose tokenizer-specific usage. The ledger is the portfolio choke point
for both resources. Successful stage debits are recorded against company and
program totals; operational usage from a failed/rolled-back attempt is retained
in the company's failed-attempt totals so retrying cannot reset either lifetime
hard wall.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

from company.models import Company, Program


class InsufficientCredits(RuntimeError):
    """Raised when a debit would exceed the company's remaining Modal credits."""


class InsufficientTokens(RuntimeError):
    """A stage/request would exceed the portable estimated/requested-token cap."""


class Ledger:
    def __init__(self, company: Company):
        self.company = company

    @property
    def credits_remaining(self) -> float:
        return (
            self.company.credit_budget
            - self.company.credit_spent
            - self.company.failed_credit_spent
        )

    @property
    def tokens_remaining(self) -> int:
        return (
            self.company.token_budget
            - self.company.token_spent
            - self.company.failed_token_spent
        )

    def can_afford(self, credits: float) -> bool:
        return credits <= self.credits_remaining + 1e-9

    def can_afford_tokens(self, tokens: int) -> bool:
        return tokens <= self.tokens_remaining

    def debit(self, program: Program, *, credits: float = 0.0, tokens: int = 0) -> None:
        """Charge a stage atomically; neither resource may cross its hard wall."""
        if (
            isinstance(credits, bool)
            or not isinstance(credits, Real)
            or not math.isfinite(float(credits))
            or credits < 0
        ):
            raise ValueError("credits must be a finite non-negative real number")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, Integral)
            or tokens < 0
        ):
            raise ValueError("tokens must be a non-negative integer")
        credits = float(credits)
        tokens = int(tokens)
        if credits > 0 and not self.can_afford(credits):
            raise InsufficientCredits(
                f"program {program.id!r} needs {credits:.1f} credits; "
                f"only {self.credits_remaining:.1f} remain"
            )
        if tokens > 0 and not self.can_afford_tokens(tokens):
            raise InsufficientTokens(
                f"program {program.id!r} needs {tokens} tokens; "
                f"only {self.tokens_remaining} remain"
            )
        self.company.credit_spent += credits
        self.company.token_spent += tokens
        program.credits_spent += credits
        program.tokens_spent += tokens
