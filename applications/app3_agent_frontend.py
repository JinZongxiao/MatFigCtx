"""
applications/app3_agent_frontend.py
应用3：作为Scientist Agent的感知前端

提供LangChain Tool接口，让Agent能够：
- 理解材料表征图像
- 结合文献语境生成可靠的科学陈述
- 返回结构化的材料知识供Agent推理
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from utils.llm_backends import make_langchain_chat_model, resolve_provider_config


# =========================
# LangChain Tool定义
# =========================
def create_matfigctx_tool(vlm_model_path: str,
                           classifier_model_path: str = None):
    """
    创建LangChain工具，供Agent调用
    """
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field
    from core.vlm_inference import load_vlm, generate_sis
    from core.subfig_classifier import classify_subfigs

    print(f"[MatFigCtx] Loading VLM: {vlm_model_path}")
    model, processor = load_vlm(vlm_model_path)

    class MatFigInput(BaseModel):
        image_path:   str           = Field(description="Path to the materials figure image")
        caption:      str           = Field(default="", description="Figure caption from the paper")
        body_text:    str           = Field(default="", description="Body text describing the figure")
        abstract:     str           = Field(default="", description="Paper abstract for global context")
        sub_images:   list          = Field(default=[], description="List of sub-figure image paths")

    def analyze_figure(image_path: str, caption: str = "",
                       body_text: str = "", abstract: str = "",
                       sub_images: list = []) -> str:
        """
        分析材料表征图像，返回科学陈述句
        """
        # 子图分类
        sub_image_type = []
        if classifier_model_path and sub_images:
            sub_image_type = classify_subfigs(sub_images, classifier_model_path)

        evidence = {
            "img_path":        image_path,
            "img_caption":     caption,
            "img_cap_text":    body_text,
            "abstract":        abstract,
            "conclusion":      "",
            "sub_images":      sub_images,
            "sub_image_type":  sub_image_type,
        }

        try:
            sis = generate_sis(evidence, model, processor)
            return json.dumps({
                "status":     "success",
                "sis":        sis,
                "fig_types":  [s["type"] for s in sub_image_type],
                "image_path": image_path,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    tool = StructuredTool(
        name="analyze_materials_figure",
        description=(
            "Analyzes a materials characterization figure (SEM, TEM, XRD, property curve, etc.) "
            "with context from the paper. Returns scientifically informative statements (SIS) "
            "grounded in the documentary evidence. Use this tool when you need to understand "
            "what a materials figure shows."
        ),
        args_schema=MatFigInput,
        func=analyze_figure,
    )
    return tool


def create_agent(vlm_model_path: str,
                 classifier_model_path: str = None,
                 cfg: dict | None = None,
                 provider: str = None):
    """
    创建完整的Materials Scientist Agent
    """
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    # 创建MatFigCtx工具
    matfig_tool = create_matfigctx_tool(vlm_model_path, classifier_model_path)

    # LLM
    cfg = cfg or {}
    llm = make_langchain_chat_model(
        cfg,
        provider=provider,
        default="ernie",
        temperature=0,
    )

    # Agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert materials scientist assistant with the ability to analyze "
         "characterization figures from research papers. You can use the analyze_materials_figure "
         "tool to understand figures and extract scientific information. "
         "Always ground your answers in the figure content and paper context. "
         "Provide specific, quantitative information when available."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [matfig_tool], prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[matfig_tool],
        verbose=True,
        max_iterations=5,
    )

    return agent_executor


def run_interactive(config_path: str = None, provider: str = None):
    """
    交互式Agent Demo
    """
    cfg = load_config(config_path or "configs/config.yaml")
    resolved = resolve_provider_config(cfg, provider=provider, default="ernie")

    agent = create_agent(
        vlm_model_path       = cfg.get("VLM_MODEL_PATH", ""),
        classifier_model_path= cfg.get("CLASSIFIER_PATH", ""),
        cfg                  = cfg,
        provider             = resolved["provider"],
    )

    print(f"\n[MatFigCtx Agent] Ready via {resolved['provider']} / {resolved['model']}. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        try:
            result = agent.invoke({"input": user_input})
            print(f"\nAgent: {result['output']}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    run_interactive(config_path="configs/config.yaml")
