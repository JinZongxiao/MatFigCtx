# MatFigCtx（中文文档）

**面向材料表征图像自动科学信息抽取的上下文约束多模态框架**

[English](README.md) | [中文](README.zh-CN.md)

---

## 概述

MatFigCtx 是一个从材料科学论文图像中抽取结构化科学信息的多模态框架。  
系统联合利用图像、图注、正文引用段落以及全局文档上下文（标题、摘要、结论），生成可追溯的科学信息陈述（SIS）。

支持四类下游应用：

| 应用 | 任务 | 关键技术 |
|-----|------|----------|
| App 1 | SIS 训练数据生成 | Qwen3-VL / InternVL3.5 + MinerU |
| App 2 | 材料知识图谱构建 | OpenAI 兼容 LLM + Neo4j |
| App 3 | 科学家 Agent 前端 | LangChain + Tool Calling |
| App 4 | Agent B 知识图谱问答 | 三元组检索 + 覆盖率约束问答 |

---

## 安装

```bash
# 1) 克隆仓库
git clone https://github.com/JinZongxiao/MatFigCtx.git
cd MatFigCtx

# 2) 创建虚拟环境（推荐）
conda create -n matfigctx python=3.10 -y
conda activate matfigctx

# 3) 安装依赖
pip install -r requirements.txt

# 4) 配置 API Key
cp .env.example .env
```

MinerU 需要单独安装：

```bash
pip install magic-pdf[full]
```

---

## 配置

在 `configs/config.yaml` 中设置数据路径、模型路径和输出目录，例如：

```yaml
PDF_INPUT_DIR:    "/path/to/raw_papers"
PAPER_OUTPUT_DIR: "/path/to/parsed_output"
ICT_OUTPUT_DIR:   "/path/to/evidence_chains"
VLM_MODEL_PATH:   "/path/to/Qwen3-VL-8B-Instruct"
CLASSIFIER_PATH:  "/path/to/image_classifier.pt"
```

`.env` 中设置密钥（不要提交到仓库）：

```bash
ERNIE_API_KEY=sk-xxxx
DEEPSEEK_API_KEY=sk-xxxx
DASHSCOPE_API_KEY=sk-xxxx
NEO4J_PASSWORD=your_password
```

---

## 用法

### App 1：SIS 训练数据生成

```bash
python pipeline.py --app 1 --pdf-dir /path/to/papers
# 若已完成解析，可跳过 PDF 解析
python pipeline.py --app 1 --skip-parse
```

### App 2：知识图谱构建

```bash
python pipeline.py --app 2
```

### App 3：科学家 Agent

```bash
python pipeline.py --app 3
```

### App 4：知识图谱问答 Agent

```bash
python pipeline.py --app 4
```

---

## Benchmark（基准评测）

```bash
# 消融实验
python -m benchmark.ablation --model qwen3-8b --setting full

# MatCha 流程
python -m benchmark.matcha_pipeline --model qwen3-8b
python -m benchmark.matcha_eval --model qwen3-8b --setting this_work

# 余弦相似度
python -m benchmark.cosine_eval

# LLM 评审
python -m benchmark.llm_judge \
  --input-csv /path/to/model_outputs.csv \
  --output-csv /path/to/judge_scores.csv

# 可视化分析
python -m benchmark.analysis
```

---

## 目录结构

请参考英文文档中的完整结构说明：`README.md`。

---

## 许可证

本项目基于 MIT License 发布，详见 `LICENSE`。
