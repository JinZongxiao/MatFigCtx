"""
pipeline.py
MatFigCtx主入口：串联完整pipeline或单独运行各应用
"""

import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MatFigCtx: Context-Grounded Materials Figure Understanding")
    parser.add_argument("--config",  default="configs/config.yaml", help="Config file path")
    parser.add_argument("--app",     choices=["all", "1", "2", "3", "4"], default="all",
                        help="Which application to run: 1=training data, 2=KG, 3=figure agent, 4=KG agent")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Skip PDF parsing (use existing w_dataset)")
    parser.add_argument("--sis-dir", default=None, help="Override SIS output directory")
    parser.add_argument("--pdf-dir", default=None, help="Override PDF input directory")
    parser.add_argument("--provider", default=None, help="OpenAI-compatible provider: ernie, deepseek, or dashscope")
    args = parser.parse_args()

    print("=" * 60)
    print("  MatFigCtx: Context-Grounded Materials Figure Understanding")
    print("=" * 60)

    if args.app in ("all", "1"):
        print("\n[App 1] Generating SIS Training Data...")
        from applications.app1_training_data import run as run_app1
        run_app1(
            config_path=args.config,
            pdf_dir=args.pdf_dir,
            output_dir=args.sis_dir,
            skip_parse=args.skip_parse,
        )

    if args.app in ("all", "2"):
        print("\n[App 2] Building Knowledge Graph...")
        from applications.app2_knowledge_graph import run as run_app2
        run_app2(
            config_path=args.config,
            sis_dir=args.sis_dir,
            provider=args.provider,
        )

    if args.app == "3":
        print("\n[App 3] Starting Scientist Agent...")
        from applications.app3_agent_frontend import run_interactive
        run_interactive(config_path=args.config, provider=args.provider)

    if args.app == "4":
        print("\n[App 4] Starting KG Agent...")
        from applications.app4_agent_b_kgqa import run_interactive
        run_interactive(config_path=args.config, provider=args.provider)

    print("\n[MatFigCtx] Done.")


if __name__ == "__main__":
    main()
