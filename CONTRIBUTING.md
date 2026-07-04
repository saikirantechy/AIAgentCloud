# Contributing to AI Agent Cloud

First off, thanks for your interest! This workshop thrives on community contributions — whether it's fixing a bug, improving the documentation, adding a new dataset challenge, or helping others run the notebook on different hardware.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Adding a New Dataset Challenge](#adding-a-new-dataset-challenge)
- [Style Guide](#style-guide)
- [Questions?](#questions)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

---

## How to Contribute

### 🐛 Report Bugs

Open an [issue](https://github.com/saikirantechy/AIAgentCloud/issues) with:

1. **A clear title** describing the problem
2. **Steps to reproduce** — which section/cell, what GPU type, exact error message
3. **Expected vs actual behavior**
4. **Environment details** — Colab vs local, Python version, torch version (`torch.__version__`)

### 💡 Suggest Enhancements

Got an idea for a new section, a better dataset, or a hardware optimization? Open an issue with the label `enhancement` and describe:

- What you'd like to add or change
- Why it's useful for the workshop
- Any implementation ideas (even rough ones)

### 📝 Improve Documentation

Typos, unclear instructions, or better examples — all welcome! See [Making Changes](#making-changes) below.

### 🧪 Add a Dataset Challenge

The workshop's Section 5 (Bring Your Own Data) is meant to grow. See [Adding a New Dataset Challenge](#adding-a-new-dataset-challenge) for guidelines.

---

## Development Setup

```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/AIAgentCloud.git
cd AIAgentCloud

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 4. Install dependencies
pip install -U transformers datasets accelerate evaluate bitsandbytes trl peft protobuf sentencepiece hf_transfer

# 5. Remove torchvision/torchaudio (avoids cryptic ABI mismatch errors)
pip uninstall -y torchvision torchaudio

# 6. Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"
```

---

## Making Changes

### Branching

Create a branch from `main` for your work:

```bash
git checkout -b fix/descriptive-name   # for bug fixes
git checkout -b feat/feature-name      # for new features
git checkout -b docs/what-you-changed  # for documentation
```

### File Structure

```
.
├── Gemma4_Finetuning_Workshop.ipynb   # Primary Colab notebook
├── gemma4_finetuning_workshop.py      # Python script (auto-generated from notebook)
├── README.md                          # Project documentation
├── CONTRIBUTING.md                    # This file
├── LICENSE                            # MIT License
└── .gitignore                         # Exclusions
```

**Note:** `gemma4_finetuning_workshop.py` is auto-generated from the Colab notebook. If you modify the notebook, you may need to update the `.py` script for consistency.

### What to Avoid

- ❌ **No hardcoded API keys or tokens** — use environment variables or Colab secrets
- ❌ **No large files** — keep the repo lean; reference datasets via Hugging Face or URLs instead
- ❌ **Don't remove error handling** — the workshop runs in live settings where things break. Every try/except block earned its keep.
- ❌ **No model weights** — never commit `.safetensors`, `.bin`, or checkpoint directories

---

## Testing

Since this is a workshop notebook, testing is mostly manual. Please:

1. **Run the notebook end-to-end** on a free Colab T4 GPU before submitting
2. Verify that all cells complete without errors (expected error messages from try/except blocks are OK)
3. Check that the loss curve converges (loss should drop measurably by step 20)
4. Confirm the before/after comparison in Section 4 shows a visible difference

### For code changes in the `.py` files:

```bash
# Syntax check
python -c "import ast; ast.parse(open('gemma4_finetuning_workshop.py').read()); print('✅ No syntax errors')"

# Quick import check (won't load the full model, just verifies imports resolve)
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from huggingface_hub import login, HfApi
print('✅ All imports resolve')
"
```

---

## Pull Request Process

1. **Ensure your branch is up to date** with `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run through the testing steps** above and include a screenshot or log output in your PR description

3. **Open a pull request** with:
   - A clear title (e.g., `fix: resolve OOM when loading E4B on T4`)
   - A description of what changed and why
   - Notes on any manual testing you did
   - Screenshots of the notebook output (if applicable)

4. **Expect review within a few days**. A maintainer may ask for changes — this is normal and welcome!

5. **Once approved**, a maintainer will merge your PR. Congratulations 🎉

---

## Adding a New Dataset Challenge

Section 5 of the workshop invites users to bring their own data. You can contribute a pre-built challenge dataset:

### Template

Add a new section to the notebook (or submit a markdown file as a guide) with:

```python
# ── Challenge: [Your Challenge Name] ─────────────────────
# Description: 1-2 sentences explaining what the model will learn
# Difficulty: beginner / intermediate / advanced

challenge_examples = [
    {
        "instruction": "...",
        "response": "..."
    },
    # ... 6-15 examples
]
```

### Guidelines

- **Keep it small** — 6–15 examples, each response under 100 tokens
- **Make the pattern obvious** — the before/after comparison should be striking
- **Stay text-only** — no images, no multi-modal (yet)
- **Avoid harmful content** — no hate speech, illegal activity, or personal data
- **Document your dataset** — what task is it teaching? What should the output look like?

### Example Challenges We'd Love to See

| Category | Idea |
|---|---|
| **Formatting** | Teach JSON output, bullet-point lists, or markdown tables |
| **Persona** | Act as a pirate, a teacher, or a customer support agent |
| **Knowledge** | Summarize a specific domain (e.g., SQL queries, Git commands) |
| **Translation** | Convert informal text to formal, or English to emoji |
| **Safety** | Recognize and refuse harmful prompts politely |

---

## Style Guide

### For Notebook Markdown Cells

- Use **ATX headings** (`##`, `###`) with a space after `#`
- Keep line length under 80 characters for readability
- Use `⚠️`, `✅`, `❌`, `➡️` emojis consistently for state indicators
- Use **bold** for UI labels like **Runtime → Restart session**
- Use `inline code` for Python identifiers, file paths, and terminal commands

### For Python Code

- Follow [PEP 8](https://peps.python.org/pep-0008/) where possible
- Use descriptive variable names (no single-letter names outside of math contexts)
- Include docstrings for any new functions
- Prefer `try/except` blocks with specific exception types over bare `except:`
- Match the existing commenting style (`# ── Section headers ──` with em-dashes)

---

## Questions?

- Open a [discussion](https://github.com/saikirantechy/AIAgentCloud/discussions)
- Reach out via the repository's GitHub Issues

**Thanks again for helping make "Why Prompting Alone Is Not Enough" better for everyone! 🚀**
