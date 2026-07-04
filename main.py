"""
AIAgentCloud — One-Stop B2B Operations Hub
=============================================
Multi-Agent Orchestrator Gateway

Three autonomous agent pillars powering finance, supply chain, and
trust/operations for micro-SMEs.

          ┌────────────────────────────────────────┐
          │       AIAgentCloud One-Stop Hub        │
          └───────────────────┬────────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Finance Agent   │  │ Supply Chain Agt │  │ Trust & Ops Agt  │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ • Invoice Mgmt   │  │ • Stock Levels   │  │ • Freelancer Esc │
│ • Payment Terms  │  │ • Pickup Logis.  │  │ • Supplier Verif │
│ • Wholesale Agri │  │ • Small Batches  │  │ • Factory Techs  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from ml_engine import (
    generate_finance,
    generate_supply_chain,
    generate_operations,
    is_available as ml_available,
)

app = FastAPI(
    title="AIAgentCloud: One-Stop B2B Operations Hub",
    description="Unified multi-agent orchestrator for finance, supply chain, "
                "and trust/operations — replacing fragmented B2B software with "
                "three autonomous AI agent pillars.",
    version="1.0.0",
)

# ───────────────────────────────────────────────────────
#  Data Models
# ───────────────────────────────────────────────────────


class AgentTaskRequest(BaseModel):
    category: str  # "finance", "supply_chain", "operations"
    problem_statement: str
    context_data: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    agents: List[str]
    ml_engine: str


class AgentAction(BaseModel):
    action: str
    detail: str


class OrchestratorResponse(BaseModel):
    agent: str
    status: str
    actions_executed: List[str]
    impact_metric: str
    follow_up: Optional[Dict[str, Any]] = None


# ───────────────────────────────────────────────────────
#  Health & Info Endpoints
# ───────────────────────────────────────────────────────


@app.get("/")
async def root():
    return {
        "service": "AIAgentCloud",
        "version": "1.0.0",
        "description": "One-Stop B2B Operations Hub",
    }


@app.get("/health")
async def health():
    ml_status = "active" if ml_available() else "fallback (static)"
    return {
        "status": "healthy",
        "agents": ["finance", "supply_chain", "operations"],
        "ml_engine": ml_status,
    }


# ───────────────────────────────────────────────────────
#  Pillar 1 — Financial Operations Agent
# ───────────────────────────────────────────────────────


def _dispatch_finance_agent(
    problem: str, context: Dict[str, Any]
) -> OrchestratorResponse:
    """
    Handles:
      - Invoice management / unstructured invoice extraction
      - Payment terms negotiation via dynamic credit profiling
      - Group-buying pooling for wholesale agricultural pricing

    Uses the ML engine for context-aware responses when available.
    """
    result = generate_finance(problem, context)
    return OrchestratorResponse(
        agent=result.agent,
        status=result.status,
        actions_executed=result.actions_executed,
        impact_metric=result.impact_metric,
        follow_up=result.follow_up if result.follow_up else None,
    )


# ───────────────────────────────────────────────────────
#  Pillar 2 — Supply Chain & Procurement Agent
# ───────────────────────────────────────────────────────


def _dispatch_supply_chain_agent(
    problem: str, context: Dict[str, Any]
) -> OrchestratorResponse:
    """
    Handles:
      - Real-time supplier stock-level lookups
      - Dependable outbound pickup logistics routing
      - Small-batch contract manufacturing matchmaking

    Uses the ML engine for context-aware responses when available.
    """
    result = generate_supply_chain(problem, context)
    return OrchestratorResponse(
        agent=result.agent,
        status=result.status,
        actions_executed=result.actions_executed,
        impact_metric=result.impact_metric,
        follow_up=result.follow_up if result.follow_up else None,
    )


# ───────────────────────────────────────────────────────
#  Pillar 3 — Trust, Verification & Dispatch Agent
# ───────────────────────────────────────────────────────


def _dispatch_operations_agent(
    problem: str, context: Dict[str, Any]
) -> OrchestratorResponse:
    """
    Handles:
      - Milestone-based escrow to prevent freelancer ghosting
      - Automated background / risk checks for new suppliers
      - On-demand emergency repair technician dispatch for factories

    Uses the ML engine for context-aware responses when available.
    """
    result = generate_operations(problem, context)
    return OrchestratorResponse(
        agent=result.agent,
        status=result.status,
        actions_executed=result.actions_executed,
        impact_metric=result.impact_metric,
        follow_up=result.follow_up if result.follow_up else None,
    )


# ───────────────────────────────────────────────────────
#  Unified Dispatch Endpoint
# ───────────────────────────────────────────────────────


@app.post(
    "/api/v1/orchestrator/dispatch",
    response_model=OrchestratorResponse,
    summary="Dispatch a B2B task to the appropriate AI agent pillar",
    description="Routes operational requests to Finance, Supply Chain, or "
                "Trust & Operations agents based on the problem category.",
)
async def orchestrate_b2b_task(request: AgentTaskRequest):
    """
    Unified entry point handling cross-functional B2B operational pains
    directly from a single control center dashboard.

    **Categories:**
    - `finance`       → Financial Operations Agent
    - `supply_chain`  → Supply Chain & Procurement Agent
    - `operations`    → Trust, Verification & Dispatch Agent
    """
    category = request.category.lower()

    if category == "finance":
        return _dispatch_finance_agent(request.problem_statement, request.context_data)

    elif category == "supply_chain":
        return _dispatch_supply_chain_agent(
            request.problem_statement, request.context_data
        )

    elif category == "operations":
        return _dispatch_operations_agent(
            request.problem_statement, request.context_data
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operational agent category '{category}'. "
                   f"Use one of: finance, supply_chain, operations.",
        )


# ───────────────────────────────────────────────────────
#  Per-Pillar Detail Endpoints (stubs)
# ───────────────────────────────────────────────────────


@app.get("/api/v1/finance/summary")
async def finance_summary():
    """Get a summary of the latest finance agent run."""
    return {
        "invoices_processed": 47,
        "payment_terms_optimized": 12,
        "group_buying_pools_active": 3,
    }


@app.get("/api/v1/supply-chain/track")
async def supply_chain_track():
    """Get real-time tracking for active supply chain dispatches."""
    return {
        "active_shipments": 8,
        "eta_range_minutes": [12, 45],
        "suppliers_contacted": 14,
    }


@app.get("/api/v1/trust/status")
async def trust_status():
    """Get verification and escrow status."""
    return {
        "verifications_pending": 3,
        "escrow_milestones_active": 6,
        "technicians_dispatched_today": 2,
    }


# ───────────────────────────────────────────────────────
#  Entry Point
# ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # set to False in production
    )
