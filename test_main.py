"""
Tests for the AIAgentCloud Multi-Agent Orchestrator.

Run with:
    pytest test_main.py -v

Or directly:
    python -m pytest test_main.py -v
"""

from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)

# ───────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────

SAMPLE_CONTEXT = {
    "vendor_id": "VENDOR-042",
    "quarter": "Q3",
}


def dispatch_request(category: str, context: dict = None):
    """Helper to send a dispatch request."""
    return client.post(
        "/api/v1/orchestrator/dispatch",
        json={
            "category": category,
            "problem_statement": "Test problem",
            "context_data": context or SAMPLE_CONTEXT,
        },
    )


# ───────────────────────────────────────────────────────
#  Health & Info
# ───────────────────────────────────────────────────────


class TestRootEndpoint:
    def test_root_returns_service_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AIAgentCloud"
        assert data["version"] == "1.0.0"
        assert "B2B" in data["description"]


class TestHealthEndpoint:
    def test_health_returns_healthy(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "finance" in data["agents"]
        assert "supply_chain" in data["agents"]
        assert "operations" in data["agents"]
        assert len(data["agents"]) == 3


# ───────────────────────────────────────────────────────
#  Finance Agent
# ───────────────────────────────────────────────────────


class TestFinanceAgentDispatch:
    def test_dispatch_finance_returns_200(self):
        response = dispatch_request("finance")
        assert response.status_code == 200

    def test_dispatch_finance_agent_name(self):
        response = dispatch_request("finance")
        assert response.json()["agent"] == "Financial_Operations_Agent"

    def test_dispatch_finance_status(self):
        response = dispatch_request("finance")
        assert response.json()["status"] == "optimized"

    def test_dispatch_finance_has_three_actions(self):
        response = dispatch_request("finance")
        actions = response.json()["actions_executed"]
        assert len(actions) == 3

    def test_dispatch_finance_actions_contain_keywords(self):
        response = dispatch_request("finance")
        actions = "\n".join(response.json()["actions_executed"])
        assert "invoice" in actions.lower()
        assert "credit" in actions.lower()
        assert "group-buying" in actions.lower()

    def test_dispatch_finance_has_impact_metric(self):
        response = dispatch_request("finance")
        metric = response.json()["impact_metric"]
        assert "hours" in metric.lower()
        assert len(metric) > 10

    def test_dispatch_finance_has_follow_up(self):
        response = dispatch_request("finance")
        follow_up = response.json()["follow_up"]
        assert follow_up is not None
        assert "recommended_next" in follow_up
        assert "savings_estimate" in follow_up


# ───────────────────────────────────────────────────────
#  Supply Chain Agent
# ───────────────────────────────────────────────────────


class TestSupplyChainAgentDispatch:
    def test_dispatch_supply_chain_returns_200(self):
        response = dispatch_request("supply_chain")
        assert response.status_code == 200

    def test_dispatch_supply_chain_agent_name(self):
        response = dispatch_request("supply_chain")
        assert response.json()["agent"] == "Supply_Chain_Logistics_Agent"

    def test_dispatch_supply_chain_status(self):
        response = dispatch_request("supply_chain")
        assert response.json()["status"] == "synchronized"

    def test_dispatch_supply_chain_has_three_actions(self):
        response = dispatch_request("supply_chain")
        assert len(response.json()["actions_executed"]) == 3

    def test_dispatch_supply_chain_actions_contain_keywords(self):
        response = dispatch_request("supply_chain")
        actions = "\n".join(response.json()["actions_executed"])
        assert "inventory" in actions.lower() or "stock" in actions.lower()
        assert "pickup" in actions.lower() or "route" in actions.lower()
        assert "manufacturing" in actions.lower()

    def test_dispatch_supply_chain_has_follow_up(self):
        response = dispatch_request("supply_chain")
        follow_up = response.json()["follow_up"]
        assert follow_up is not None
        assert "eta_minutes" in follow_up


# ───────────────────────────────────────────────────────
#  Operations (Trust & Dispatch) Agent
# ───────────────────────────────────────────────────────


class TestOperationsAgentDispatch:
    def test_dispatch_operations_returns_200(self):
        response = dispatch_request("operations")
        assert response.status_code == 200

    def test_dispatch_operations_agent_name(self):
        response = dispatch_request("operations")
        assert response.json()["agent"] == "Trust_and_Dispatch_Agent"

    def test_dispatch_operations_status(self):
        response = dispatch_request("operations")
        assert response.json()["status"] == "deployed"

    def test_dispatch_operations_has_three_actions(self):
        response = dispatch_request("operations")
        assert len(response.json()["actions_executed"]) == 3

    def test_dispatch_operations_actions_contain_keywords(self):
        response = dispatch_request("operations")
        actions = "\n".join(response.json()["actions_executed"])
        assert "milestone" in actions.lower() or "escrow" in actions.lower()
        assert "verification" in actions.lower() or "background" in actions.lower()
        assert "technician" in actions.lower() or "repair" in actions.lower()

    def test_dispatch_operations_has_follow_up(self):
        response = dispatch_request("operations")
        follow_up = response.json()["follow_up"]
        assert follow_up is not None
        assert "dispatch_eta" in follow_up


# ───────────────────────────────────────────────────────
#  Edge Cases & Error Handling
# ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unknown_category_returns_400(self):
        response = dispatch_request("inventory")
        assert response.status_code == 400

    def test_unknown_category_error_message(self):
        response = dispatch_request("inventory")
        detail = response.json()["detail"]
        assert "inventory" in detail
        assert "finance" in detail
        assert "supply_chain" in detail

    def test_case_insensitivity_finance(self):
        """Category matching should be case-insensitive."""
        for cat in ["Finance", "FINANCE", "finance"]:
            response = dispatch_request(cat)
            assert response.status_code == 200, f"Failed for category '{cat}'"
            assert response.json()["agent"] == "Financial_Operations_Agent"

    def test_case_insensitivity_supply_chain(self):
        for cat in ["Supply_Chain", "SUPPLY_CHAIN", "Supply_chain"]:
            response = dispatch_request(cat)
            assert response.status_code == 200, f"Failed for category '{cat}'"

    def test_case_insensitivity_operations(self):
        for cat in ["Operations", "OPERATIONS", "Operations"]:
            response = dispatch_request(cat)
            assert response.status_code == 200, f"Failed for category '{cat}'"

    def test_empty_context_data(self):
        """Dispatch should work even with empty context data."""
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            json={
                "category": "finance",
                "problem_statement": "Test",
                "context_data": {},
            },
        )
        assert response.status_code == 200

    def test_missing_context_data(self):
        """Dispatch should fail if context_data is missing."""
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            json={
                "category": "finance",
                "problem_statement": "Test",
            },
        )
        assert response.status_code == 422  # validation error

    def test_missing_problem_statement(self):
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            json={
                "category": "finance",
                "context_data": {},
            },
        )
        assert response.status_code == 422

    def test_empty_category_returns_400(self):
        """Empty string is a valid string — routes as unknown category → 400."""
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            json={
                "category": "",
                "problem_statement": "Test",
                "context_data": {},
            },
        )
        assert response.status_code == 400


# ───────────────────────────────────────────────────────
#  Detail Endpoints
# ───────────────────────────────────────────────────────


class TestDetailEndpoints:
    def test_finance_summary(self):
        response = client.get("/api/v1/finance/summary")
        assert response.status_code == 200
        data = response.json()
        assert "invoices_processed" in data
        assert "payment_terms_optimized" in data
        assert "group_buying_pools_active" in data

    def test_supply_chain_track(self):
        response = client.get("/api/v1/supply-chain/track")
        assert response.status_code == 200
        data = response.json()
        assert "active_shipments" in data
        assert "eta_range_minutes" in data
        assert "suppliers_contacted" in data

    def test_trust_status(self):
        response = client.get("/api/v1/trust/status")
        assert response.status_code == 200
        data = response.json()
        assert "verifications_pending" in data
        assert "escrow_milestones_active" in data
        assert "technicians_dispatched_today" in data


# ───────────────────────────────────────────────────────
#  Response Model Integrity
# ───────────────────────────────────────────────────────


class TestResponseModel:
    def test_all_dispatch_responses_match_orchestrator_schema(self):
        """All three agent responses should have the same required fields."""
        for category in ["finance", "supply_chain", "operations"]:
            response = dispatch_request(category)
            data = response.json()
            assert "agent" in data
            assert "status" in data
            assert "actions_executed" in data
            assert "impact_metric" in data
            # follow_up is optional but currently always present

    def test_actions_executed_is_list_of_strings(self):
        for category in ["finance", "supply_chain", "operations"]:
            response = dispatch_request(category)
            actions = response.json()["actions_executed"]
            assert isinstance(actions, list)
            for action in actions:
                assert isinstance(action, str)
                assert len(action) > 5  # no empty strings

    def test_http_method_not_allowed(self):
        """GET on the dispatch endpoint should fail."""
        response = client.get("/api/v1/orchestrator/dispatch")
        assert response.status_code == 405  # Method Not Allowed

    def test_post_on_root_returns_405(self):
        response = client.post("/")
        assert response.status_code == 405

    def test_invalid_json_body_returns_422(self):
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            data="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
