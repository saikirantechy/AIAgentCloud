# ☁️ AIAgentCloud — One-Stop B2B Operations Hub

### *Replace five fragmented tools with three autonomous AI agent pillars*

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/google/gemma-4-E2B-it)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://github.com/saikirantechy/AIAgentCloud/blob/main/LICENSE)

---

## 📋 Overview

**AIAgentCloud** is a unified platform that replaces the fragmented software stack B2B businesses juggle daily — one for billing, one for shipping, one for contracting — with a **centralized multi-agent orchestrator**.

The platform comprises two integrated layers:

| Layer | What it does |
|---|---|
| **🤖 Multi-Agent Orchestrator** (`main.py`) | Routes B2B operational requests across three autonomous AI agent pillars: Finance, Supply Chain, and Trust & Operations |
| **🧠 Fine-Tuning Engine** (`Gemma4_Finetuning_Workshop.ipynb`) | Uses Google's Gemma 4 with memory-optimized LoRA fine-tuning to power the intelligence behind each agent pillar |

---

## 🏗️ Architecture: The 3 Agent Pillars

```
                  ┌────────────────────────────────────────┐
                  │       AIAgentCloud One-Stop Hub        │
                  └───────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Finance Agent   │        │ Supply Chain Agt │        │ Trust & Ops Agt  │
├──────────────────┤        ├──────────────────┤        ├──────────────────┤
│ • Invoice Mgmt   │        │ • Stock Levels   │        │ • Freelancer Esc │
│ • Payment Terms  │        │ • Pickup Logis.  │        │ • Supplier Verif │
│ • Wholesale Agri │        │ • Small Batches  │        │ • Factory Techs  │
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

### Pillar 1 — Financial Operations Agent
Automates the financial backbone of any B2B operation:

- **Invoice Management** — Extracts line items from unstructured invoice metadata
- **Payment Terms Negotiation** — Generates dynamic credit profiles to optimize terms
- **Wholesale Agricultural Pooling** — Enables group-buying for small restaurants to access wholesale pricing

### Pillar 2 — Supply Chain & Procurement Agent
Keeps inventory and logistics moving without manual coordination:

- **Real-Time Stock Lookups** — Queries supplier inventory caches for live availability
- **Outbound Pickup Logistics** — Routes dependable third-party pickup fleets automatically
- **Small-Batch Manufacturing** — Matches contract manufacturing runs to available capacity

### Pillar 3 — Trust, Verification & Dispatch Agent
Handles the human side of B2B operations — trust, payments, and emergencies:

- **Milestone Escrow** — Prevents freelancer ghosting with smart milestone-based payouts
- **Supplier Background Checks** — Automated risk verification for new supplier entities
- **Emergency Technician Dispatch** — Scans regional networks for certified industrial repair technicians

---

## 🚀 Quick Start

### Run the Orchestrator

```bash
# Clone the repo
git clone https://github.com/saikirantechy/AIAgentCloud.git
cd AIAgentCloud

# Install dependencies
pip install -r requirements.txt

# Start the API gateway
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Try the Dispatch Endpoint

```bash
# Dispatch a finance task
curl -X POST http://localhost:8000/api/v1/orchestrator/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "category": "finance",
    "problem_statement": "Process outstanding invoices for Q3",
    "context_data": {"vendor_id": "VENDOR-042", "quarter": "Q3"}
  }'

# Dispatch a supply chain task
curl -X POST http://localhost:8000/api/v1/orchestrator/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "category": "supply_chain",
    "problem_statement": "Check stock for raw material order",
    "context_data": {"sku": "RM-1080", "quantity": 500}
  }'

# Dispatch an operations task
curl -X POST http://localhost:8000/api/v1/orchestrator/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "category": "operations",
    "problem_statement": "Verify new supplier and set up escrow",
    "context_data": {"supplier_id": "SUPP-203", "contract_value": 15000}
  }'
```

### API Health Check

```bash
curl http://localhost:8000/health
# → {"status": "healthy", "agents": ["finance", "supply_chain", "operations"]}
```

---

## 📦 Repository Contents

| File | Description |
|---|---|
| `main.py` | 🏗️ **FastAPI Multi-Agent Orchestrator** — the three-pillar B2B operations gateway |
| `Gemma4_Finetuning_Workshop.ipynb` | 🧠 **Fine-Tuning Engine** — memory-optimized Gemma 4 LoRA workshop for Colab |
| `gemma4_finetuning_workshop.py` | Python script version of the fine-tuning workshop |
| `requirements.txt` | Python package dependencies |
| `README.md` | This file |
| `CONTRIBUTING.md` | Contribution guidelines |
| `LICENSE` | MIT License |
| `.gitignore` | Repository exclusions |

---

## 🧠 The Fine-Tuning Engine

Every AI agent pillar is powered by fine-tuned LLM capabilities. The [`Gemma4_Finetuning_Workshop.ipynb`](Gemma4_Finetuning_Workshop.ipynb) is a complete hands-on workshop that demonstrates **fine-tuning Google's Gemma 4 models on consumer hardware** (free Colab T4 GPU).

### Key features of the workshop:

- ✅ Loads **Gemma 4 (2B)** in **4-bit quantization** (~5 GB VRAM)
- ✅ **Memory-optimized training** — custom LoRA prep avoids the common 8.7 GiB OOM spike
- ✅ **3–5 minute training** on a free T4 GPU
- ✅ **Before/after comparison** proving fine-tuning beats prompting alone
- ✅ **Bring Your Own Data** challenge in Section 5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_6ss4EShsd2hNOW6kVznnvySIxtNeqJd)

---

## 🛠️ API Reference

### `POST /api/v1/orchestrator/dispatch`

The unified entry point. Routes a B2B problem to the correct agent pillar.

**Request body:**

```json
{
  "category": "finance | supply_chain | operations",
  "problem_statement": "Description of the task",
  "context_data": { ... }
}
```

**Response:**

```json
{
  "agent": "Financial_Operations_Agent",
  "status": "optimized",
  "actions_executed": [
    "Extracted line items from unstructured invoice metadata.",
    "Generated structured dynamic credit profile..."
  ],
  "impact_metric": "Eliminated 10+ hours of manual administrative overhead.",
  "follow_up": { ... }
}
```

### Detail Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Service info and version |
| `GET /health` | Agent health status |
| `GET /api/v1/finance/summary` | Finance agent metrics |
| `GET /api/v1/supply-chain/track` | Active shipment tracking |
| `GET /api/v1/trust/status` | Verification and escrow status |

---

## 🧪 Demo / Pitch Sequence

When presenting AIAgentCloud, run through this 3-step demo:

```
1. Health Check  →  curl /health
   → Shows all 3 agents are live and ready

2. Finance Dispatch  →  curl ... -d '{"category": "finance", ...}'
   → Agent extracts invoice data, optimizes payment terms

3. Cross-Pillar Flow
   → finance → supply_chain → operations
   → Demonstrates how the same platform handles billing,
     logistics, and trust in under 2 seconds each
```

### One-Liner Pitch

> *"B2B business owners are forced to juggle five different fragmented software tools just to run daily business tasks — one for billing, one for shipping, and one for contracting. AIAgentCloud replaces this entirely with a centralized backend that deploys specialized AI agents that talk to each other. Whether it's clearing an invoice backlog, matching a supply log, or verifying a supplier background, our platform serves as the single operational operating system for micro-SMEs."*

---

## ⚙️ Workshop Sections Walkthrough

### Section 0 — Environment Setup
| Subsection | What happens |
|---|---|
| **0.1** | GPU detection |
| **0.2** | Install pinned dependencies — **must restart runtime after this** |
| **0.3** | Model tier picker (E2B / E4B / 12B) + Hugging Face auth |
| **0.4** | 🔁 Recovery cell — clears GPU memory if anything crashes |

### Section 1 — Load Gemma 4 in 4-bit
Loads the model via `BitsAndBytesConfig` with NF4 quantization. Sanity check answers "What is LoRA fine-tuning?".

### Section 2 — Prepare the Dataset
Teaches a **"Workshop-Bot house style"** (📍 formatting) via 6 example pairs × 6 repeats.

### Section 3 — Configure LoRA and Train
- **LoRA:** r=16, alpha=32, all-linear targets
- **Training:** batch_size=1, gradient_accumulation=8, max_steps=20
- **Custom memory optimization** upcasts only norm layers (not embeddings), avoiding a common 8.7 GiB OOM

### Section 4 — Merge & Compare
Merges adapter, reloads in bf16, runs side-by-side comparison:

```
Prompt: What is the capital of Germany?
  After fine-tuning: The capital of Germany is **Berlin**.

Prompt: What's 5 + 7?
  After fine-tuning: 5 + 7 = **12**
```

### Section 5 — 🧪 Bring Your Own Data Challenge
Swap in your own instruction/response pairs and prove the model learned *your* pattern.

### Section 6 — Troubleshooting
Quick-reference table for the 9 most common errors.

---

## 🐛 Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ImportError` / `AttributeError` after installs | Runtime wasn't restarted | `Runtime → Restart session`, continue from 0.3 |
| `RuntimeError: operator torchvision::nms does not exist` | torch/torchvision ABI mismatch | `pip uninstall torchvision torchaudio`, restart runtime |
| `CUDA out of memory` | Model too big / fragmented memory | Run Recovery cell (0.4), use E2B tier |
| 403 / gated repo error | License not accepted | Visit model page → click "Agree and access repository" |
| `HF_TOKEN` not found | Secret not configured | 🔑 key → add `HF_TOKEN` → enable notebook access |
| Training hangs >2 min | First run — model caching to disk | Check `nvidia-smi` for GPU activity |

---

## 🤗 Publishing to the Hugging Face Hub

After fine-tuning, you can share your model with the community. There are **two approaches**:

### Push the LoRA Adapter Only (~16 MB)
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

YOUR_USER = "your-hf-username"
REPO_NAME = "gemma4-finetuned-adapter"

trainer.model.save_pretrained(f"./{REPO_NAME}")
tokenizer.save_pretrained(f"./{REPO_NAME}")
trainer.model.push_to_hub(f"{YOUR_USER}/{REPO_NAME}")
tokenizer.push_to_hub(f"{YOUR_USER}/{REPO_NAME}")
```

### Push the Merged Model (~3–6 GB)
```python
merged_model.push_to_hub(f"{YOUR_USER}/gemma4-finetuned-merged")
tokenizer.push_to_hub(f"{YOUR_USER}/gemma4-finetuned-merged")
```

See the full guide with loading, private repos, and update examples in the [Gemma 4 workshop notebook](Gemma4_Finetuning_Workshop.ipynb).

---

## 🚀 Going Further at Home

- **Scale the orchestrator** — add real ML inference endpoints behind each agent pillar
- **Use a real dataset** — swap the 6 demo examples for hundreds/thousands of instruction pairs
- **Train longer** — raise `max_steps` from 20 to 500–2000 and add an eval loop
- **Try Unsloth** — [unsloth.ai](https://unsloth.ai) reduces VRAM by 50–80% and trains 2x faster
- **Export to GGUF** — serve locally with [Ollama](https://ollama.ai) or [llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Add auth & rate limiting** — secure the orchestrator with API keys and throttling

---

## 🙏 Acknowledgments

- **Google** for the Gemma 4 model family
- **Hugging Face** for Transformers, PEFT, TRL, and the Hub
- **Colab** for providing free T4 GPUs for accessible fine-tuning
- **bitsandbytes** for 4-bit quantization support
- **FastAPI** for the async Python web framework

---

## 📄 License

This project is licensed under the MIT License. The underlying Gemma 4 models are subject to Google's [Gemma License](https://ai.google.dev/gemma/terms).

---

<div align="center">
  
**Built with ❤️ for the B2B community — because fragmented software is not enough.**

[Report Issue](https://github.com/saikirantechy/AIAgentCloud/issues) · [Star on GitHub](https://github.com/saikirantechy/AIAgentCloud) · [Contributions Welcome](CONTRIBUTING.md)

</div>
