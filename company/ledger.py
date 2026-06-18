"""Dual budget ledger: LLM tokens (reasoning) + Modal GPU credits (compute).

Both are *real* resources (spec §6) — the ledger is the single choke point that
makes the CEO's prioritization decisions have teeth. Debits are recorded against
both the company totals and the owning program.
"""
from __future__ import annotations

from company.models import Company, Program


class InsufficientCredits(RuntimeError):
    """Raised when a debit would exceed the company's remaining Modal credits."""


class Ledger:
    def __init__(self, company: Company):
        self.company = company

    @property
    def credits_remaining(self) -> float:
        return self.company.credit_budget - self.company.credit_spent

    @property
    def tokens_remaining(self) -> int:
        return self.company.token_budget - self.company.token_spent

    def can_afford(self, credits: float) -> bool:
        return credits <= self.credits_remaining + 1e-9

    def debit(self, program: Program, *, credits: float = 0.0, tokens: int = 0) -> None:
        """Charge a stage/experiment to a program. Raises if credits would go negative.

        Tokens are allowed to overspend their (soft) budget — we never want a
        reasoning call to hard-fail mid-stage — but Modal credits are a hard wall
        because they are literal money.
        """
        if credits > 0 and not self.can_afford(credits):
            raise InsufficientCredits(
                f"program {program.id!r} needs {credits:.1f} credits; "
                f"only {self.credits_remaining:.1f} remain"
            )
        self.company.credit_spent += credits
        self.company.token_spent += tokens
        program.credits_spent += credits
        program.tokens_spent += tokens
