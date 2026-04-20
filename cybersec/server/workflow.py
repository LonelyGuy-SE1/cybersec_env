"""Enterprise workflow actor for CybersecEnv (ticketing and approvals)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from .world_state import ControlAction

Urgency = Literal["low", "normal", "high"]
TicketStatus = Literal[
    "pending_review",
    "approved_waiting",
    "ready",
    "rejected",
    "executed",
]


@dataclass
class TicketRecord:
    ticket_id: str
    requested_action: ControlAction
    target: str
    urgency: Urgency
    status: TicketStatus
    created_step: int
    review_due_step: int
    execute_ready_step: Optional[int]
    decision_note: str = ""


@dataclass
class WorkflowUpdate:
    ticket_id: str
    status: TicketStatus
    message: str


class WorkflowEngine:
    """Models enterprise change-control delays and approval dynamics."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._tickets: Dict[str, TicketRecord] = {}
        self._counter = 0

    def open_ticket(
        self,
        *,
        tick: int,
        requested_action: ControlAction,
        target: str,
        urgency: Urgency,
        target_evidence: float,
    ) -> TicketRecord:
        self._counter += 1
        ticket_id = f"TKT-{self._counter:04d}"

        review_delay = _review_delay(urgency)
        review_due_step = tick + review_delay
        execute_ready_step: Optional[int] = None
        status: TicketStatus = "pending_review"

        auto_approval = target_evidence >= 0.70 and urgency == "high"
        if auto_approval:
            status = "approved_waiting"
            execute_ready_step = review_due_step + 1

        ticket = TicketRecord(
            ticket_id=ticket_id,
            requested_action=requested_action,
            target=target,
            urgency=urgency,
            status=status,
            created_step=tick,
            review_due_step=review_due_step,
            execute_ready_step=execute_ready_step,
            decision_note="queued for security and operations review",
        )
        self._tickets[ticket_id] = ticket
        return ticket

    def update(
        self, *, tick: int, evidence_by_target: Dict[str, float]
    ) -> List[WorkflowUpdate]:
        updates: List[WorkflowUpdate] = []

        for ticket in self._tickets.values():
            if ticket.status in {"executed", "rejected", "ready"}:
                continue

            if ticket.status == "pending_review" and tick >= ticket.review_due_step:
                evidence = float(evidence_by_target.get(ticket.target, 0.0))
                approval_prob = _approval_probability(
                    action=ticket.requested_action,
                    urgency=ticket.urgency,
                    evidence=evidence,
                )
                approved = self._rng.random() <= approval_prob
                if approved:
                    ticket.status = "approved_waiting"
                    ticket.execute_ready_step = tick + _execution_delay(ticket.urgency)
                    ticket.decision_note = f"Approved with evidence={evidence:.2f}; awaiting execution window"
                    updates.append(
                        WorkflowUpdate(
                            ticket_id=ticket.ticket_id,
                            status=ticket.status,
                            message=(
                                f"Ticket {ticket.ticket_id} approved; execution available"
                                f" at step {ticket.execute_ready_step}"
                            ),
                        )
                    )
                else:
                    ticket.status = "rejected"
                    ticket.decision_note = f"Rejected due to insufficient confidence (evidence={evidence:.2f})"
                    updates.append(
                        WorkflowUpdate(
                            ticket_id=ticket.ticket_id,
                            status=ticket.status,
                            message=(
                                f"Ticket {ticket.ticket_id} rejected by enterprise"
                                " change board"
                            ),
                        )
                    )
                continue

            if (
                ticket.status == "approved_waiting"
                and ticket.execute_ready_step is not None
                and tick >= ticket.execute_ready_step
            ):
                ticket.status = "ready"
                updates.append(
                    WorkflowUpdate(
                        ticket_id=ticket.ticket_id,
                        status=ticket.status,
                        message=(
                            f"Ticket {ticket.ticket_id} is ready for EXECUTE_TICKET"
                        ),
                    )
                )

        return updates

    def get_ticket(self, ticket_id: str) -> Optional[TicketRecord]:
        return self._tickets.get(ticket_id)

    def mark_executed(self, ticket_id: str, note: str) -> Optional[TicketRecord]:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return None
        ticket.status = "executed"
        ticket.decision_note = note
        return ticket

    def visible_tickets(self) -> List[TicketRecord]:
        tickets = list(self._tickets.values())
        tickets.sort(key=lambda t: t.created_step)
        return tickets

    def ready_ticket_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                ticket.ticket_id
                for ticket in self._tickets.values()
                if ticket.status == "ready"
            )
        )


def _review_delay(urgency: Urgency) -> int:
    if urgency == "high":
        return 1
    if urgency == "normal":
        return 2
    return 3


def _execution_delay(urgency: Urgency) -> int:
    if urgency == "high":
        return 1
    if urgency == "normal":
        return 2
    return 3


def _approval_probability(
    *, action: ControlAction, urgency: Urgency, evidence: float
) -> float:
    base = 0.35
    base += 0.50 * evidence

    if urgency == "high":
        base += 0.10
    elif urgency == "low":
        base -= 0.08

    if action in {"BLOCK_EGRESS", "ISOLATE_ASSET"}:
        base -= 0.05
    if action == "PATCH_ASSET":
        base += 0.04

    if evidence >= 0.85:
        base += 0.08

    if base < 0.05:
        return 0.05
    if base > 0.98:
        return 0.98
    return float(base)
