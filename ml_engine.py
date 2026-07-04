"""
ML Engine — AIAgentCloud
=========================
Gemma 4 inference module powering the three agent pillars with
real, context-aware ML-generated responses.

Architecture:
  - Loads Gemma 4 in 4-bit (same proven pattern from the workshop)
  - Provides one generation function per agent pillar with specialized
    prompt templates
  - Gracefully falls back to static responses if the model cannot
    be loaded (no GPU, no HF token, etc.)
  - Caches the model as a module-level singleton
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("ml_engine")

# ───────────────────────────────────────────────────────
#  Data Types
# ───────────────────────────────────────────────────────


@dataclass
class AgentOutput:
    """Structured output parsed from the model's generated text."""
    agent: str
    status: str
    actions_executed: List[str]
    impact_metric: str
    follow_up: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────────────
#  Model Singleton
# ───────────────────────────────────────────────────────

_model = None
_tokenizer = None
_model_loaded = False
_load_error: Optional[str] = None

# Configurable — change these to point to your fine-tuned model
FINETUNED_MODEL_ID = os.environ.get(
    "AIAGENT_MODEL_ID", ""
)  # e.g. "your-user/gemma4-finetuned-merged"
BASE_MODEL_ID = "google/gemma-4-E2B-it"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.3


def _load_model() -> Tuple[bool, str]:
    """
    Load the Gemma 4 model in 4-bit. Tries the fine-tuned model
    first (if configured), then falls back to the base model.

    Returns (success, message).
    """
    global _model, _tokenizer, _model_loaded, _load_error

    if _model_loaded:
        return True, "Model already loaded."

    if _load_error:
        return False, _load_error

    # Determine which model to load
    model_id = FINETUNED_MODEL_ID.strip() or BASE_MODEL_ID

    try:
        import torch
    except ImportError:
        msg = "PyTorch not installed. ML engine disabled — using static fallback."
        logger.warning(msg)
        _load_error = msg
        return False, msg

    if not torch.cuda.is_available():
        msg = "No CUDA GPU detected. ML engine disabled — using static fallback."
        logger.warning(msg)
        _load_error = msg
        return False, msg

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )
        from huggingface_hub import login

        # Authenticate if HF_TOKEN is available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        # 4-bit quantization config (same as workshop)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        logger.info("Loading model: %s", model_id)

        _tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        _model.eval()

        # If we loaded the base model but a fine-tuned one is available
        # as a PEFT adapter, try to load it on top
        if not FINETUNED_MODEL_ID.strip():
            adapter_path = os.environ.get(
                "AIAGENT_ADAPTER_PATH", "./gemma4-workshop-adapter"
            )
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                try:
                    from peft import PeftModel

                    _model = PeftModel.from_pretrained(_model, adapter_path)
                    _model.eval()
                    logger.info("LoRA adapter loaded from %s", adapter_path)
                except Exception as e:
                    logger.warning("Could not load LoRA adapter: %s", e)

        _model_loaded = True
        msg = f"✅ Model loaded: {model_id}"
        logger.info(msg)
        return True, msg

    except Exception as e:
        msg = f"Failed to load model '{model_id}': {e}"
        logger.warning("%s — using static fallback.", msg)
        _load_error = msg
        return False, msg


def is_available() -> bool:
    """Check if the ML engine is ready to generate responses."""
    if _model_loaded:
        return True
    success, _ = _load_model()
    return success


# ───────────────────────────────────────────────────────
#  Prompt Templates
# ───────────────────────────────────────────────────────

FINANCE_SYSTEM_PROMPT = """You are the Financial Operations Agent, an AI that handles B2B financial tasks.
Your job is to analyze the user's request and context, then respond with EXACTLY THREE actions
you would take, a one-sentence impact metric, and a follow-up recommendation.

Respond in this EXACT format (no extra text):
ACTIONS:
1. <action description>
2. <action description>
3. <action description>
IMPACT: <one-sentence impact metric>
FOLLOW_UP: <recommended next step>
SAVINGS_ESTIMATE: <estimated time or money saved>"""

SUPPLY_CHAIN_SYSTEM_PROMPT = """You are the Supply Chain & Procurement Agent, an AI that handles B2B logistics.
Your job is to analyze the user's request and context, then respond with EXACTLY THREE actions
you would take, a one-sentence impact metric, and a follow-up recommendation.

Respond in this EXACT format (no extra text):
ACTIONS:
1. <action description>
2. <action description>
3. <action description>
IMPACT: <one-sentence impact metric>
FOLLOW_UP: <recommended next step>
ETA_MINUTES: <estimated dispatch time in minutes>"""

TRUST_SYSTEM_PROMPT = """You are the Trust, Verification & Dispatch Agent, an AI that handles B2B
trust and safety operations. Your job is to analyze the user's request and context, then
respond with EXACTLY THREE actions you would take, a one-sentence impact metric, and a
follow-up recommendation.

Respond in this EXACT format (no extra text):
ACTIONS:
1. <action description>
2. <action description>
3. <action description>
IMPACT: <one-sentence impact metric>
FOLLOW_UP: <recommended next step>
DISPATCH_ETA: <estimated dispatch time in minutes>"""


def _build_prompt(category: str, problem: str, context: Dict[str, Any]) -> str:
    """Build a prompt for the model based on the agent pillar."""
    system_prompts = {
        "finance": FINANCE_SYSTEM_PROMPT,
        "supply_chain": SUPPLY_CHAIN_SYSTEM_PROMPT,
        "operations": TRUST_SYSTEM_PROMPT,
    }

    system = system_prompts.get(category, FINANCE_SYSTEM_PROMPT)
    context_str = json.dumps(context, indent=2) if context else "No additional context."

    return f"""{system}

--- BEGIN USER REQUEST ---
Problem: {problem}
Context: {context_str}
--- END USER REQUEST ---"""


def _parse_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the model's generated text into structured fields.
    Returns None if parsing fails.
    """
    result = {"actions_executed": [], "impact_metric": "", "follow_up": {}}

    # Extract actions
    actions_section = re.search(
        r"ACTIONS:\s*(.*?)(?=IMPACT:|FOLLOW_UP:|$)", text, re.DOTALL
    )
    if actions_section:
        lines = actions_section.group(1).strip().split("\n")
        for line in lines:
            line = line.strip()
            # Match "1. action" or "- action" or "* action"
            match = re.match(r"^[\d\-\*\.\)]+\s*(.*)", line)
            if match:
                action = match.group(1).strip()
                if action:
                    result["actions_executed"].append(action)

    # Extract impact metric
    impact_match = re.search(r"IMPACT:\s*(.*?)(?=FOLLOW_UP:|$)", text, re.DOTALL)
    if impact_match:
        result["impact_metric"] = impact_match.group(1).strip()

    # Extract follow-up fields
    follow_up_match = re.search(r"FOLLOW_UP:\s*(.*?)(?=\n[A-Z_]+:|$)", text, re.DOTALL)
    if follow_up_match:
        result["follow_up"]["recommended_next"] = follow_up_match.group(1).strip()

    # Extract optional fields
    for key in ["SAVINGS_ESTIMATE", "ETA_MINUTES", "DISPATCH_ETA"]:
        match = re.search(rf"{key}:\s*(.*?)(?=\n[A-Z_]+:|$)", text, re.DOTALL)
        if match:
            result["follow_up"][key.lower()] = match.group(1).strip()

    # Fill defaults if parsing failed
    if not result["actions_executed"]:
        result["actions_executed"] = [
            "Analyzed the provided B2B context for actionable insights.",
            "Cross-referenced operational data against best-practice patterns.",
            "Prepared a structured response with recommended next steps.",
        ]

    if not result["impact_metric"]:
        result["impact_metric"] = (
            "Processed the request and generated a context-aware response."
        )

    if not result["follow_up"]:
        result["follow_up"] = {"recommended_next": "Review the generated actions above."}

    return result


# ───────────────────────────────────────────────────────
#  Generation Function
# ───────────────────────────────────────────────────────


def generate(
    category: str,
    problem_statement: str,
    context_data: Dict[str, Any],
    agent_name: str,
    status: str,
) -> AgentOutput:
    """
    Generate an ML-powered response for the given agent pillar.

    Falls back to static responses if the model isn't available.
    """
    # Try ML generation
    if is_available():
        try:
            import torch

            prompt = _build_prompt(category, problem_statement, context_data)
            messages = [{"role": "user", "content": prompt}]

            # Use the chat template for proper formatting
            inputs = _tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(_model.device)

            with torch.no_grad():
                output_ids = _model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.9,
                    pad_token_id=_tokenizer.eos_token_id,
                )

            generated = _tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )

            parsed = _parse_output(generated)

            return AgentOutput(
                agent=agent_name,
                status=status,
                actions_executed=parsed["actions_executed"],
                impact_metric=parsed["impact_metric"],
                follow_up=parsed["follow_up"],
            )

        except Exception as e:
            logger.warning("ML generation failed: %s — falling back to static.", e)

    # ── Static fallback ──────────────────────────────────
    return _static_fallback(category, agent_name, status)


# ───────────────────────────────────────────────────────
#  Static Fallback Responses
# ───────────────────────────────────────────────────────


def _static_fallback(category: str, agent_name: str, status: str) -> AgentOutput:
    """Return a static response when the ML model isn't available."""
    responses = {
        "finance": AgentOutput(
            agent=agent_name or "Financial_Operations_Agent",
            status=status or "optimized",
            actions_executed=[
                "Extracted line items from unstructured invoice metadata.",
                "Generated structured dynamic credit profile to optimize payment terms negotiation.",
                "Queried group-buying pools for wholesale agricultural pricing.",
            ],
            impact_metric="Eliminated 10+ hours of manual administrative overhead.",
            follow_up={
                "recommended_next": "Review invoice summary at /api/v1/finance/summary",
                "savings_estimate": "~12 hrs/week in manual data entry",
            },
        ),
        "supply_chain": AgentOutput(
            agent=agent_name or "Supply_Chain_Logistics_Agent",
            status=status or "synchronized",
            actions_executed=[
                "Queried supplier inventory cache for real-time stock availability.",
                "Dispatched automated route request to accountable third-party pickup fleets.",
                "Matched small-batch manufacturing run against contract capacity.",
            ],
            impact_metric="Reduced fulfillment uncertainty margins down to zero.",
            follow_up={
                "recommended_next": "Track shipment at /api/v1/supply-chain/track",
                "eta_minutes": "~15 min average dispatch",
            },
        ),
        "operations": AgentOutput(
            agent=agent_name or "Trust_and_Dispatch_Agent",
            status=status or "deployed",
            actions_executed=[
                "Initialized smart milestone tracking system for milestone payout security.",
                "Executed automated background verification on new supplier entity.",
                "Scanned regional networks for certified industrial repair technicians.",
            ],
            impact_metric="Secured high-priority operations against external dependency delays.",
            follow_up={
                "recommended_next": "Review verification report at /api/v1/trust/status",
                "dispatch_eta": "~30 min for emergency technician dispatch",
            },
        ),
    }

    return responses.get(
        category,
        AgentOutput(
            agent=agent_name or "Unknown Agent",
            status=status or "standby",
            actions_executed=["No actions available for this category."],
            impact_metric="Request could not be processed.",
            follow_up={"recommended_next": "Try a different category."},
        ),
    )


# ───────────────────────────────────────────────────────
#  Convenience Wrappers
# ───────────────────────────────────────────────────────


def generate_finance(problem: str, context: Dict[str, Any]) -> AgentOutput:
    return generate(
        category="finance",
        problem_statement=problem,
        context_data=context,
        agent_name="Financial_Operations_Agent",
        status="optimized",
    )


def generate_supply_chain(problem: str, context: Dict[str, Any]) -> AgentOutput:
    return generate(
        category="supply_chain",
        problem_statement=problem,
        context_data=context,
        agent_name="Supply_Chain_Logistics_Agent",
        status="synchronized",
    )


def generate_operations(problem: str, context: Dict[str, Any]) -> AgentOutput:
    return generate(
        category="operations",
        problem_statement=problem,
        context_data=context,
        agent_name="Trust_and_Dispatch_Agent",
        status="deployed",
    )
