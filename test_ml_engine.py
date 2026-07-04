"""
Tests for the ML Engine module.

Run with:
    pytest test_ml_engine.py -v

Note: Tests that require the actual ML model will gracefully skip
if no GPU is available. The static fallback and parsing tests run
everywhere.
"""

import json
import pytest
from ml_engine import (
    AgentOutput,
    _parse_output,
    _static_fallback,
    _build_prompt,
    generate,
    generate_finance,
    generate_supply_chain,
    generate_operations,
    is_available,
)

# ───────────────────────────────────────────────────────
#  AgentOutput Dataclass
# ───────────────────────────────────────────────────────


class TestAgentOutput:
    def test_default_follow_up_is_empty_dict(self):
        output = AgentOutput(
            agent="Test",
            status="ok",
            actions_executed=["Action 1"],
            impact_metric="Good.",
        )
        assert output.follow_up == {}

    def test_can_set_follow_up(self):
        output = AgentOutput(
            agent="Test",
            status="ok",
            actions_executed=["Action 1"],
            impact_metric="Good.",
            follow_up={"key": "value"},
        )
        assert output.follow_up["key"] == "value"


# ───────────────────────────────────────────────────────
#  Static Fallback
# ───────────────────────────────────────────────────────


class TestStaticFallback:
    def test_finance_fallback(self):
        result = _static_fallback("finance", "Financial_Ops_Agent", "optimized")
        assert result.agent == "Financial_Ops_Agent"
        assert result.status == "optimized"
        assert len(result.actions_executed) == 3
        assert any("invoice" in a.lower() for a in result.actions_executed)
        assert "hours" in result.impact_metric.lower()

    def test_supply_chain_fallback(self):
        result = _static_fallback(
            "supply_chain", "Supply_Chain_Agent", "synchronized"
        )
        assert result.agent == "Supply_Chain_Agent"
        assert result.status == "synchronized"
        assert len(result.actions_executed) == 3
        assert any("inventory" in a.lower() for a in result.actions_executed)
        assert "fulfillment" in result.impact_metric.lower()

    def test_operations_fallback(self):
        result = _static_fallback("operations", "Trust_Agent", "deployed")
        assert result.agent == "Trust_Agent"
        assert result.status == "deployed"
        assert len(result.actions_executed) == 3
        assert any("milestone" in a.lower() for a in result.actions_executed)

    def test_unknown_category_fallback(self):
        result = _static_fallback("unknown", "Agent", "standby")
        assert result.status == "standby"
        assert "No actions" in result.actions_executed[0]

    def test_default_agent_name_when_none(self):
        result = _static_fallback("finance", None, None)
        assert result.agent == "Financial_Operations_Agent"
        assert result.status == "optimized"


# ───────────────────────────────────────────────────────
#  Prompt Building
# ───────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_finance_prompt_contains_system_text(self):
        prompt = _build_prompt(
            "finance",
            "Process invoice INV-123",
            {"vendor": "ACME", "amount": 5000},
        )
        assert "Financial Operations Agent" in prompt
        assert "INV-123" in prompt
        assert "ACME" in prompt

    def test_supply_chain_prompt_contains_system_text(self):
        prompt = _build_prompt(
            "supply_chain",
            "Check stock for SKU-456",
            {"warehouse": "WH-1"},
        )
        assert "Supply Chain" in prompt
        assert "SKU-456" in prompt

    def test_operations_prompt_contains_system_text(self):
        prompt = _build_prompt(
            "operations",
            "Verify supplier SUPP-789",
            {"contract_value": 10000},
        )
        assert "Trust" in prompt or "Verification" in prompt
        assert "SUPP-789" in prompt

    def test_empty_context_included(self):
        prompt = _build_prompt("finance", "Test problem", {})
        assert "Test problem" in prompt
        assert "No additional context" in prompt or "{}" in prompt

    def test_context_is_json_serialized(self):
        prompt = _build_prompt(
            "finance", "Test", {"nested": {"key": "value"}, "list": [1, 2]}
        )
        assert "nested" in prompt
        assert "list" in prompt


# ───────────────────────────────────────────────────────
#  Output Parsing
# ───────────────────────────────────────────────────────


class TestParseOutput:
    def test_parse_actions_impact_follow_up(self):
        text = """ACTIONS:
1. Extracted line items from invoice.
2. Generated credit profile.
3. Queried group-buying pools.
IMPACT: Reduced manual work by 12 hours per week.
FOLLOW_UP: Review invoice summary.
SAVINGS_ESTIMATE: ~$500/month"""
        result = _parse_output(text)
        assert result is not None
        assert len(result["actions_executed"]) == 3
        assert "Extracted" in result["actions_executed"][0]
        assert "hours" in result["impact_metric"]
        assert result["follow_up"]["recommended_next"] == "Review invoice summary."
        assert result["follow_up"]["savings_estimate"] == "~$500/month"

    def test_parse_with_dash_list(self):
        text = """ACTIONS:
- Checked inventory levels
- Dispatched pickup fleet
- Matched manufacturing capacity
IMPACT: Fulfillment uncertainty eliminated."""
        result = _parse_output(text)
        assert len(result["actions_executed"]) == 3
        assert "inventory" in result["actions_executed"][0].lower()

    def test_parse_missing_actions_uses_defaults(self):
        text = """Just some random text without the right format."""
        result = _parse_output(text)
        assert result is not None
        assert len(result["actions_executed"]) == 3
        assert "Analyzed" in result["actions_executed"][0]
        assert result["impact_metric"] != ""

    def test_parse_partial_output(self):
        text = """ACTIONS:
1. Only one action here.
IMPACT: Some metric."""
        result = _parse_output(text)
        assert len(result["actions_executed"]) == 1

    def test_parse_follow_up_with_colon(self):
        text = """ACTIONS:
1. Action one.
IMPACT: Metric.
FOLLOW_UP: Check the dashboard for details.
DISPATCH_ETA: ~25 min"""
        result = _parse_output(text)
        assert result["follow_up"]["recommended_next"] == "Check the dashboard for details."
        assert result["follow_up"]["dispatch_eta"] == "~25 min"


# ───────────────────────────────────────────────────────
#  Generate Function (static fallback path — no GPU)
# ───────────────────────────────────────────────────────


class TestGenerateFallback:
    """These tests exercise the static fallback path (no GPU available in CI)."""

    def test_generate_finance_fallback(self):
        result = generate_finance(
            "Process overdue invoices",
            {"vendor_id": "V001", "amount": 15000},
        )
        assert result.agent == "Financial_Operations_Agent"
        assert result.status == "optimized"
        assert len(result.actions_executed) >= 1
        assert len(result.impact_metric) > 5

    def test_generate_supply_chain_fallback(self):
        result = generate_supply_chain(
            "Replenish stock for SKU-789",
            {"warehouse": "WH-3", "reorder_point": 100},
        )
        assert result.agent == "Supply_Chain_Logistics_Agent"
        assert result.status == "synchronized"
        assert len(result.actions_executed) >= 1

    def test_generate_operations_fallback(self):
        result = generate_operations(
            "Verify new supplier",
            {"supplier_name": "NewCo", "country": "US"},
        )
        assert result.agent == "Trust_and_Dispatch_Agent"
        assert result.status == "deployed"
        assert len(result.actions_executed) >= 1

    def test_generate_all_return_same_schema(self):
        """All three convenience wrappers return AgentOutput with same fields."""
        finance = generate_finance("Test", {})
        supply = generate_supply_chain("Test", {})
        ops = generate_operations("Test", {})

        for result in [finance, supply, ops]:
            assert hasattr(result, "agent")
            assert hasattr(result, "status")
            assert hasattr(result, "actions_executed")
            assert hasattr(result, "impact_metric")
            assert hasattr(result, "follow_up")

    def test_generate_with_context_does_not_error(self):
        """Ensure context_data doesn't cause errors even in fallback."""
        result = generate(
            category="finance",
            problem_statement="Handle invoice backlog",
            context_data={"count": 47, "total": 84200.0},
            agent_name="FinanceAgent",
            status="optimized",
        )
        assert result.agent == "FinanceAgent"
        assert result.status == "optimized"
        assert len(result.actions_executed) >= 1


# ───────────────────────────────────────────────────────
#  Integration: main.py dispatch uses ml_engine
# ───────────────────────────────────────────────────────


class TestIntegrationWithMain:
    """Verify main.py's dispatch endpoints use the ML engine."""

    def test_finance_dispatch_returns_valid_response(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            json={
                "category": "finance",
                "problem_statement": "Process invoices for Q3 from vendor VENDOR-042",
                "context_data": {"vendor_id": "VENDOR-042", "quarter": "Q3", "total": 84200},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "Financial_Operations_Agent"
        assert data["status"] == "optimized"
        assert len(data["actions_executed"]) >= 1
        assert len(data["impact_metric"]) > 5

    def test_supply_chain_dispatch_returns_valid_response(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            json={
                "category": "supply_chain",
                "problem_statement": "Check stock levels for raw materials",
                "context_data": {"sku": "RM-1080", "warehouse": "WH-1"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "Supply_Chain_Logistics_Agent"
        assert data["status"] == "synchronized"
        assert len(data["actions_executed"]) >= 1

    def test_operations_dispatch_returns_valid_response(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        response = client.post(
            "/api/v1/orchestrator/dispatch",
            json={
                "category": "operations",
                "problem_statement": "Run background check on new supplier SUPP-203",
                "context_data": {"supplier_id": "SUPP-203", "country": "US"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "Trust_and_Dispatch_Agent"
        assert data["status"] == "deployed"
        assert len(data["actions_executed"]) >= 1

    def test_health_includes_ml_engine_status(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "ml_engine" in data
        assert "active" in data["ml_engine"] or "fallback" in data["ml_engine"]
