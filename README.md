# 🔧 Fine-Tuning Gemma 4 on Consumer Hardware

### *Why Prompting Alone Is Not Enough — A Live Workshop*

[![Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1OqX7FFDkzI2xH-q-zNQJLq0H4Bj7iLdn#scrollTo=AdK4gpC-5rWH)
[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/google/gemma-4-E2B-it)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://github.com/saikirantechy/AIAgentCloud/blob/main/LICENSE)

---

## 📋 Overview

This repository contains a complete workshop for **fine-tuning Google's Gemma 4 models** on consumer-grade hardware — specifically a **free Google Colab T4 GPU (15 GB VRAM)**. 

The workshop demonstrates that **prompting alone is not enough**: to truly customize a model's behavior, style, and knowledge, you need fine-tuning. By the end, you'll have:

- ✅ Loaded **Gemma 4 (2B)** in **4-bit quantization** (~5 GB VRAM)
- ✅ Configured **LoRA adapters** for parameter-efficient fine-tuning
- ✅ Trained on a custom dataset in **3–5 minutes**
- ✅ Merged the adapter and run a **before/after comparison**
- ✅ Learned how to **swap in your own data** for custom tasks

---

## 🎯 What Makes This Unique

| Feature | Detail |
|---|---|
| **Consumer hardware** | Runs on a **free T4 GPU** (no Colab Pro needed) |
| **Live-workshop hardened** | Tested with 50+ concurrent users on shared venue wifi |
| **Memory-optimized** | Custom `prepare_model_for_lora_training` avoids the 8.7 GiB OOM spike that plagues PEFT's standard method |
| **Step-capped for demo** | Training completes in ~3–5 minutes, not hours |
| **Built-in BYOD challenge** | Section 5 lets you plug in your own data and prove the model learned *your* pattern |

---

## 📦 Repository Contents

| File | Description |
|---|---|
| `Gemma4_Finetuning_Workshop.ipynb` | Jupyter notebook — the primary workshop material, designed for Google Colab |
| `gemma4_finetuning_workshop.py` | Python script version of the same workshop (runnable locally or in Colab) |
| `README.md` | This file |

---

## 🧠 Prerequisites

### Hardware
- **GPU with ≥ 8 GB VRAM** (free Colab T4 works perfectly)
- For the default **E2B (2B)** model: ~5 GB VRAM in 4-bit
- For **E4B** or **12B**: Colab Pro (A100/L4) recommended

### Software
- Python 3.10+
- PyTorch (CUDA-enabled)
- See `Section 0.2` for all dependency versions

### Accounts
1. 🤗 **Hugging Face account** — [sign up free](https://huggingface.co/join)
2. Accept the **Gemma 4 license** at [huggingface.co/google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
3. Generate a **read-only token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Quick Start

### Option A: Run in Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_6ss4EShsd2hNOW6kVznnvySIxtNeqJd)

1. Click the badge above to open the notebook
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run cells **top to bottom, once each**
4. When prompted, restart the runtime after Section 0.2
5. Set your `HF_TOKEN` using the 🔑 key icon in Colab's left sidebar

### Option B: Run Locally

```bash
# Clone the repo
git clone https://github.com/saikirantechy/AIAgentCloud.git
cd AIAgentCloud

# Set your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# Install dependencies
pip install -q -U transformers datasets accelerate evaluate bitsandbytes trl peft protobuf sentencepiece hf_transfer

# Run the script
python gemma4_finetuning_workshop.py
```

---

## 📖 Workshop Sections Walkthrough

### Section 0 — Environment Setup
| Subsection | What happens |
|---|---|
| **0.1** | GPU detection — confirms you have a CUDA-capable GPU |
| **0.2** | Installs pinned dependencies (transformers, PEFT, TRL, bitsandbytes...) — **must restart runtime after this** |
| **0.3** | Configuration picker (E2B / E4B / 12B) + Hugging Face authentication |
| **0.4** | 🔁 **Recovery cell** — clears GPU memory if anything crashes |

> ⚠️ **Critical Rule:** After Section 0.2 finishes installing, you **MUST restart the runtime** (Runtime → Restart session). Skipping this is the #1 cause of errors.

### Section 1 — Load Gemma 4 in 4-bit
- Loads the model using `BitsAndBytesConfig` with NF4 quantization
- Double quantization saves additional memory
- Runs a sanity check: asks the model "What is LoRA fine-tuning?"
- Reports free GPU memory after loading

### Section 2 — Prepare the Dataset
- A small demo dataset (6 examples × 6 repeats = 36 total)
- Teaches the model a **"Workshop-Bot house style"** (📍 formatting)
- 85/15 train/eval split

### Section 3 — Configure LoRA and Train
- **LoRA config:** r=16, alpha=32, all-linear targets
- **Training:** batch_size=1, gradient_accumulation=8, max_steps=20
- **Custom `prepare_model_for_lora_training`** — upcasts only **norm layers** to fp32 (not embeddings), avoiding a common 8.7 GiB OOM
- Training completes in ~3–5 minutes on a T4
- Final loss typically drops from ~8.7 → ~0.09

### Section 4 — Merge & Compare
- Saves the LoRA adapter separately
- Merges adapter into base weights
- Reloads in bf16 for stable inference
- Runs side-by-side comparison on unseen prompts:
  - *Before:* generic response
  - *After:* responses formatted with the learned 📍 style

**Example output (from the workshop run):**
```
Prompt: What is the capital of Germany?
  After fine-tuning: The capital of Germany is **Berlin**.

Prompt: What's 5 + 7?
  After fine-tuning: 5 + 7 = **12**

Prompt: Name a popular cloud provider.
  After fine-tuning: A popular cloud provider is **Amazon Web Services (AWS)**.
```

The model's fine-tuned responses are more structured and confident. With a larger dataset or more steps, the 📍 formatting style from the training data would transfer more consistently to unseen prompts.

### Section 5 — 🧪 Bring Your Own Data Challenge
- Edit the `my_examples` list with 5–15 of your own instruction/response pairs
- The notebook trains a fresh model on your data
- Run the before/after comparison to prove the model learned *your* pattern

### Section 6 — Troubleshooting + Next Steps
- Quick-reference table for the 9 most common errors
- Guidance on scaling up: more steps, real datasets, Unsloth, GGUF export

---

## ⚙️ Model Tiers

| Tier | Model ID | Params | VRAM (4-bit) | GPU Required |
|---|---|---|---|---|
| **E2B** (default) | `google/gemma-4-E2B-it` | 2B | ~5 GB | Free T4 ✅ |
| **E4B** | `google/gemma-4-E4B-it` | 4B | ~8 GB | T4 (tight), A100/L4 ✅ |
| **12B** | `google/gemma-4-12B-it` | 12B | ~14 GB | A100/L4 (Colab Pro) |

Change `MODEL_TIER` in Section 0.3 to switch between them.

---

## 🛠️ Technical Highlights

### Memory Optimization (Why This Doesn't OOM)

The standard `peft.prepare_model_for_kbit_training()` upcasts **all** non-4-bit parameters to fp32 — including the embedding and LM head layers. For Gemma's ~256k-token vocabulary, that's a single ~8.75 GiB allocation, which instantly OOMs a T4.

**Our fix:** A lightweight replacement that only upcasts **norm layers** (layer-norm, RMS-norm) to fp32 — which is all you actually need for training stability. Embeddings stay at bf16 since they're frozen lookup tables during LoRA.

### Recovery Pattern
Every major section has a try/except block with clear error messages. The Recovery cell (0.4) clears GPU memory without restarting the runtime, letting you recover from most errors in seconds rather than minutes.

---

## 🐛 Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ImportError` / `AttributeError` after installs | Runtime wasn't restarted | `Runtime → Restart session`, continue from 0.3 |
| `RuntimeError: operator torchvision::nms does not exist` | torch/torchvision ABI mismatch | `pip uninstall torchvision torchaudio`, restart runtime |
| `CUDA out of memory` (8.7 GiB) | Embedding layer upcast | Already fixed in this notebook |
| `CUDA out of memory` (general) | Model too big / fragmented memory | Run Recovery cell (0.4), use E2B tier |
| 403 / gated repo error | License not accepted | Visit model page → click "Agree and access repository" |
| `HF_TOKEN` not found | Secret not configured | 🔑 key → add `HF_TOKEN` → enable notebook access |
| Training hangs >2 min | First run — model caching to disk | Check `nvidia-smi` for GPU activity; be patient |
| Session disconnect | Colab idle timeout / wifi drop | Reconnect → Restart session → run top to bottom |

---

## 🚀 Going Further at Home

- **Use a real dataset** — swap the 6 demo examples for hundreds/thousands of instruction pairs
- **Train longer** — raise `max_steps` from 20 to 500–2000 and add an eval loop
- **Try Unsloth** — [unsloth.ai](https://unsloth.ai) reduces VRAM by 50–80% and trains 2x faster
- **Export to GGUF** — convert the merged model for local serving with [Ollama](https://ollama.ai) or [llama.cpp](https://github.com/ggerganov/llama.cpp)
## 🤗 Publishing to the Hugging Face Hub

After fine-tuning, you can share your model with the community by pushing it to the Hugging Face Hub. There are **two approaches**, depending on whether you saved the LoRA adapter or the full merged model.

> **Note:** Pushing the merged model (~3–6 GB) takes longer and requires more disk space. The adapter-only approach (~16 MB) is much faster and sufficient for most sharing purposes.

### 1. Authenticate

First, log in to Hugging Face from your environment. You'll need a **write-access token** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```python
from huggingface_hub import login

# Option A: Login with your token directly
login(token="hf_your_write_token_here")

# Option B: Use environment variable (more secure)
# export HF_TOKEN="hf_your_write_token_here"
```

Or from the command line:
```bash
huggingface-cli login --token hf_your_write_token_here
```

### 2a. Push the LoRA Adapter Only (Recommended — ~16 MB)

This is fast, takes almost no storage, and others can load the adapter on top of the same base model.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your Hugging Face username
YOUR_USER = "your-hf-username"
REPO_NAME = "gemma4-finetuned-adapter"  # change this as you like

# Save and push the adapter
trainer.model.save_pretrained(f"./{REPO_NAME}")
tokenizer.save_pretrained(f"./{REPO_NAME}")

# Push to the Hub
trainer.model.push_to_hub(f"{YOUR_USER}/{REPO_NAME}")
tokenizer.push_to_hub(f"{YOUR_USER}/{REPO_NAME}")

print(f"✅ Adapter pushed to https://huggingface.co/{YOUR_USER}/{REPO_NAME}")
```

### 2b. Push the Merged Model (~3–6 GB)

This creates a standalone model that doesn't require the base model at inference time.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

YOUR_USER = "your-hf-username"
REPO_NAME = "gemma4-finetuned-merged"

# merge_and_unload() must have been called first (already done in Section 4)
merged_model.push_to_hub(f"{YOUR_USER}/{REPO_NAME}")
tokenizer.push_to_hub(f"{YOUR_USER}/{REPO_NAME}")

print(f"✅ Merged model pushed to https://huggingface.co/{YOUR_USER}/{REPO_NAME}")
```

### 3. Loading from the Hub

**Loading the LoRA adapter (needs the base Gemma 4 model):**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "google/gemma-4-E2B-it"
adapter_id = "your-hf-username/gemma4-finetuned-adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

**Loading the standalone merged model (no base model needed):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your-hf-username/gemma4-finetuned-merged"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

### 4. Creating a Private Repository

If you want to keep your model private during development:

```python
from huggingface_hub import HfApi

api = HfApi()
# Create a private repo
api.create_repo(
    repo_id=f"{YOUR_USER}/gemma4-private-finetune",
    private=True,
    exist_ok=True,
)
# Then push as usual with .push_to_hub()
```

### 5. Updating an Existing Repository

```python
# Push with a commit message
trainer.model.push_to_hub(
    f"{YOUR_USER}/{REPO_NAME}",
    commit_message="Retrain with 2x more data — loss improved from 0.09 to 0.04",
)
```

### Tips

- **Adapters are tiny** (~16 MB for rank-16 LoRA) — prefer pushing adapters over merged models for quick iteration
- **You need write permission** on the target repo — use a write-capable token (not read-only)
- **First push creates the repo** automatically on the Hub
- **View your models** at [huggingface.co/YOUR_USER](https://huggingface.co/)
- **Try chat template fine-tuning** — experiment with multi-turn conversation data instead of single-turn instruction/response pairs

---

## 🙏 Acknowledgments

- **Google** for the Gemma 4 model family
- **Hugging Face** for Transformers, PEFT, TRL, and the Hub
- **Colab** for providing free T4 GPUs that make this workshop accessible to everyone
- **bitsandbytes** for 4-bit quantization support

---

## 📄 License

This project is licensed under the MIT License. The underlying Gemma 4 models are subject to Google's [Gemma License](https://ai.google.dev/gemma/terms).

---

<div align="center">
  
**Built with ❤️ for the AI community — because prompting alone is not enough.**

[Report Issue](https://github.com/saikirantechy/AIAgentCloud/issues) · [Star on GitHub](https://github.com/saikirantechy/AIAgentCloud) · [Contributions Welcome](https://github.com/saikirantechy/AIAgentCloud/pulls)

</div>
