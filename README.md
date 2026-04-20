# MatFigCtx

**Context-Grounded Multimodal Framework for Automated Scientific Information Extraction from Materials Characterization Figures**

[English](README.md) | [中文](README.zh-CN.md)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

---

## Overview

MatFigCtx is a context-grounded multimodal framework that extracts structured scientific information from materials characterization figures in research papers. By jointly leveraging figure images, captions, in-text references, and global document context (title, abstract, conclusions), it generates high-quality Scientific Information Statements (SIS) grounded in documentary evidence.

**Four downstream applications are supported:**

| App | Task | Key Technology |
|-----|------|---------------|
| 1 | SIS Training Data Generation | Qwen3-VL / InternVL3.5 + MinerU |
| 2 | Materials Knowledge Graph Construction | OpenAI-compatible LLM + Neo4j |
| 3 | Scientist Agent Frontend | LangChain + Tool Calling |
| 4 | Agent B KG-QA Frontend | Triple retrieval + coverage-constrained QA |

---

## Framework Pipeline

```
PDF Papers
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: Document Parsing  (MinerU)            │
│  PDF → Markdown + structured content list       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Evidence Chain Extraction             │
│  ├─ Figure image + caption                      │
│  ├─ In-text paragraphs referencing the figure   │
│  ├─ Sub-figure segmentation  (OpenCV)           │
│  ├─ Sub-figure classification  (ResNet-50)      │
│  └─ Global context  (abstract + conclusions)    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: VLM Inference  (Qwen3-VL / InternVL) │
│  Evidence Chain → Scientific Information        │
│  Statements (SIS)  — one sentence per line      │
└────────────────────┬────────────────────────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
     App 2: KG           App 3: Agent
   SIS → Triples      LangChain Tool
   → Neo4j             → QA Frontend
```

---

## Repository Structure

```
MatFigCtx/
├── core/
│   ├── pdf_parser.py           # PDF parsing via MinerU
│   ├── subfig_segmentor.py     # Sub-figure segmentation (OpenCV)
│   ├── subfig_classifier.py    # Sub-figure classification (ResNet-50, 3-class)
│   ├── evidence_extractor.py   # Evidence chain assembly
│   └── vlm_inference.py        # SIS generation (Qwen3-VL)
│
├── applications/
│   ├── app1_training_data.py   # App 1: SIS training data generation
│   ├── app2_knowledge_graph.py # App 2: Triple extraction + Neo4j ingestion
│   ├── app3_agent_frontend.py  # App 3: LangChain figure agent
│   ├── app4_agent_b_kgqa.py    # App 4: KG retrieval + coverage-aware QA
│   └── agent_a_pdf_parser.py   # Agent A / admin-node orchestration
│
├── benchmark/
│   ├── _vlm_utils.py           # Shared VLM loader/runner (Qwen3-VL & InternVL3.5)
│   ├── ablation.py             # 7-setting × 6-model ablation study
│   ├── cosine_eval.py          # SciBERT cosine similarity evaluation
│   ├── matcha_pipeline.py      # Evidence chain generation on MatCha dataset
│   ├── matcha_eval.py          # MatCha VQA evaluation (zero-shot / this work)
│   └── analysis.py             # Result aggregation, heatmaps, ablation plots
│
├── utils/
│   └── config.py               # YAML config loader + .env secret management
│
├── configs/
│   └── config.yaml             # Unified configuration (paths, models, hyper-params)
│
├── pipeline.py                 # Top-level entry point
├── .env.example                # API key template (copy → .env, never commit)
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/MatFigCtx.git
cd MatFigCtx

# 2. Create a virtual environment (recommended)
conda create -n matfigctx python=3.10 -y
conda activate matfigctx

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and fill in your keys
```

### MinerU (PDF Parser)

MinerU requires a separate installation step. Follow the official guide:
[https://github.com/opendatalab/MinerU](https://github.com/opendatalab/MinerU)

```bash
pip install magic-pdf[full]
```

---

## Configuration

Edit `configs/config.yaml` to set model paths and data directories:

```yaml
# Core data paths
PDF_INPUT_DIR:    "/path/to/raw_papers"
PAPER_OUTPUT_DIR: "/path/to/parsed_output"
ICT_OUTPUT_DIR:   "/path/to/evidence_chains"

# VLM model paths
VLM_MODEL_PATH: "/path/to/Qwen3-VL-8B-Instruct"

# Sub-figure classifier checkpoint
CLASSIFIER_PATH: "/path/to/image_classifier.pt"

# VLM registry for benchmark (model-key → local path)
VLM_REGISTRY:
  qwen3-8b:    "/path/to/Qwen3-VL-8B-Instruct"
  internvl-8b: "/path/to/InternVL3.5-8B"
  # ... add more models as needed
```

API keys are stored in `.env` (never in the YAML):

```bash
ERNIE_API_KEY=sk-xxxx
DEEPSEEK_API_KEY=sk-xxxx
DASHSCOPE_API_KEY=sk-xxxx
NEO4J_PASSWORD=your_password
```

`DEFAULT_LLM_PROVIDER` in `configs/config.yaml` controls which OpenAI-compatible
backend is used by App 2, App 3, App 4, and `benchmark/llm_judge.py`.

---

## Usage

### App 1 — SIS Training Data Generation

Runs the full pipeline: PDF → evidence chain → SIS JSON.

```bash
# Full pipeline (includes PDF parsing)
python pipeline.py --app 1 --pdf-dir /path/to/papers

# Skip parsing if w_dataset already exists
python pipeline.py --app 1 --skip-parse
```

Output: per-paper `*_sis.json` + merged `all_training_data.json` in `SIS_OUTPUT_DIR`.

### App 2 — Knowledge Graph Construction

Extracts (subject, predicate, object) triples from SIS and ingests them into Neo4j.

```bash
python pipeline.py --app 2
```

Requires a running Neo4j instance and an OpenAI-compatible API key such as `ERNIE_API_KEY` in `.env`.

### App 3 — Scientist Agent

Interactive materials scientist agent with figure-understanding capability.

```bash
python pipeline.py --app 3
```

### App 4 — Agent B Knowledge-Graph QA

Interactive question answering over extracted triples with explicit coverage reporting.

```bash
python pipeline.py --app 4
```

---

## Benchmark

### Ablation Study

Seven evidence settings × six VLMs. Evaluates mean cosine similarity of generated SIS against reference `img_cap_text`.

**Settings:**

| Setting | Image | Caption + Body Text | Sub-fig Type | Global Context |
|---------|:-----:|:-------------------:|:------------:|:--------------:|
| `zero_shot` | ✓ | | | |
| `img_only` | ✓ | | | |
| `txt_only` | | ✓ | | |
| `full` | ✓ | ✓ | ✓ | ✓ |
| `wo_subfig_type` | ✓ | ✓ | | ✓ |
| `wo_local` | ✓ | | ✓ | ✓ |
| `wo_global` | ✓ | ✓ | ✓ | |

```bash
# Run a single combination
python -m benchmark.ablation --model qwen3-8b --setting full

# Available models: qwen3-2b, qwen3-4b, qwen3-8b, internvl-2b, internvl-4b, internvl-8b
# Available settings: zero_shot, img_only, txt_only, full, wo_subfig_type, wo_local, wo_global
```

### MatCha VQA Evaluation

```bash
# Step 1 — generate evidence chain output on MatCha figures
python -m benchmark.matcha_pipeline --model qwen3-8b

# Step 2 — VQA evaluation
python -m benchmark.matcha_eval --model qwen3-8b --setting this_work
python -m benchmark.matcha_eval --model qwen3-8b --setting zero_shot
```

### Cosine Similarity Scoring

Computes SciBERT-based cosine similarity for all model × setting combinations.

### LLM-as-Judge

Use an OpenAI-compatible judge model to score model outputs on scientific specificity,
visual description quality, and factual plausibility.

```bash
python -m benchmark.llm_judge \
  --input-csv /path/to/model_outputs.csv \
  --output-csv /path/to/judge_scores.csv
```

```bash
python -m benchmark.cosine_eval
```

### Result Visualization

Generates accuracy bar charts, cosine heatmaps, and ablation line plots.

```bash
python -m benchmark.analysis
# Outputs saved to ANALYSIS_OUTPUT_DIR (see config.yaml)
```

---

## Models

| Component | Model | Notes |
|-----------|-------|-------|
| VLM (main) | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | BF16, flash-attention-2 |
| VLM (benchmark) | Qwen3-VL {2B/4B/8B}, InternVL3.5 {2B/4B/8B} | 6 models total |
| Sub-fig classifier | ResNet-50 (fine-tuned) | 3 classes: line_plot / xrd / microstructure |
| Text encoder | [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) | Cosine similarity evaluation |
| LLM (triple extraction) | DeepSeek-Chat | App 2, via OpenAI-compatible API |
| PDF parser | [MinerU](https://github.com/opendatalab/MinerU) | Structured PDF → Markdown |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{matfigctx2025,
  title   = {},
  author  = {},
  journal = {arXiv preprint},
  year    = {},
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
