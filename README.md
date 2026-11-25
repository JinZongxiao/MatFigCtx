# MatFigCtx: Document-Grounded Evidence Extraction from Materials Figures

This repository contains the implementation of MatFigCtx for extracting traceable scientific statements from materials characterization figures in research articles.

Data can be found at Zenodo (https://doi.org/10.5281/zenodo.24457846).


## Installation

1. Clone this repository:

```bash
git clone https://github.com/JinZongxiao/MatFigCtx.git
cd MatFigCtx
```

2. Create an environment and install the required packages (tested on Linux with Python 3.10, PyTorch 2.5, and CUDA 12.1):

```bash
conda create -n matfigctx python=3.10 -y
conda activate matfigctx
pip install -r requirements.txt
pip install magic-pdf[full]
```

3. Configure paths in `configs/config.yaml`, and copy `.env.example` to `.env` if using API models:

```bash
cp .env.example .env
```

## Usage

### SIS extraction

`pipeline.py` runs batch processing from raw PDFs (or pre-parsed evidence chains) to three-stage SIS JSON files.

```bash
python pipeline.py --config configs/config.yaml --pdf-dir <pdf_dir> --out-dir <sis_out_dir>
```

Where:
- **pdf_dir**: Directory containing source PDF articles.
- **sis_out_dir**: Directory where per-article and merged SIS JSON files are saved.
- Add `--skip-parse --skip-evidence` to run directly on existing evidence chains, and `--verify` to enable multimodal visual filtering (Sec. 2.4).

### Connecting custom local or API models

MatFigCtx is not tied to specific model weights or API providers:

1. **Local / Hugging Face VLMs**: pass any local directory or Hugging Face repo ID compatible with `AutoModelForImageTextToText` and `AutoProcessor` (e.g., Qwen-VL, InternVL, LLaVA-OneVision, Llama-Vision, Pixtral, Gemma):
   ```bash
   python pipeline.py --model /path/to/your-hf-vlm --skip-parse
   ```
2. **API-based VLMs for SIS extraction**: any OpenAI-compatible multimodal endpoint (OpenAI GPT-4o, Gemini, DashScope Qwen-VL, OpenRouter, or self-hosted vLLM / SGLang) can be used directly by setting `VLM_API_BASE_URL` and `VLM_API_KEY` in `.env` and prefixing the model name with `api:`:
   ```bash
   python pipeline.py --model api:gpt-4o --skip-parse --skip-evidence
   ```
3. **API models for visual verification, KG extraction, and judging**: `core/visual_verifier.py`, `downstream/kg_extract.py`, and `eval/gold_judge.py` use a standard OpenAI-compatible client configured via `.env` (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). Switch providers by updating `.env` (e.g., `https://api.openai.com/v1`, `https://generativelanguage.googleapis.com/v1beta/openai/`, `https://dashscope.aliyuncs.com/compatible-mode/v1`, `https://api.deepseek.com`, or `http://localhost:8000/v1`).
4. **Custom Python VLM wrapper**: for non-standard local models, add a branch in `load_vlm()` and `_run_vlm()` inside `core/vlm_inference.py`.

### Downstream reuse

Scripts in `downstream/` cover knowledge graph construction, retrieval-augmented QA (Sec. 2.4), and LoRA continual pre-training (Sec. 2.5) using the input files deposited on Zenodo:

```bash
# Knowledge graph triple extraction
python -m downstream.kg_extract --sis-json <all_sis.json> --out <kg_triples.json>

# BM25 RAG question answering (arms Q0-Q5)
python -m downstream.rag_qa --ict-dir <ict_dir> --questions <odsw_qa180.json> --sis <sis.jsonl> --model <vlm_path> --out-dir <qa_out_dir>

# Gemma 4 E2B LoRA training and held-out conclusion NLL
python -m downstream.cpt_gemma train --model <gemma_path> --train-json <train_R4.json> --out-dir <lora_out_dir>
python -m downstream.cpt_gemma eval-nll --model <gemma_path> --lora-dir <lora_out_dir>/final --conclusions-json <heldout_conclusions.json> --out-csv <nll.csv>
```

### Benchmark evaluation

Scripts in `eval/` reproduce the gold-standard recall evaluation (Sec. 2.2), MatCha open-book controls and answer-leakage audit (Sec. 2.6), and MatViX comparison (Supplementary Note 8):

```bash
python -m eval.gold_judge --gold-json <gold_facts.json> --pred-json <all_sis.json> --out <judge.json>

python -m eval.matcha_eval eval --model <vlm_path> --model-tag qwen3-8b --matcha-jsonl <matcha.jsonl> --img-root <img_dir> --refs-json <refs.json> --out-dir <out_dir>
python -m eval.matcha_eval leakage --matcha-jsonl <matcha.jsonl> --refs-json <refs.json> --out <leakage.json>

python -m eval.matvix_score --eval-dir <matvix_eval_dir> --true-dir <matvix_test_dir> --ids <ids.json> --self-test --fix-keys --fix-match --fix-order
```


## Citation

Citation details will be added upon publication.

## License

This project is licensed under the [MIT License](LICENSE).
