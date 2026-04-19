"""
benchmark/analysis.py
汇总分析脚本：生成 MatCha 正确率对比、余弦相似度热力图、消融折线图。

Usage:
    python -m benchmark.analysis
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config


MODEL_ORDER = ["qwen3-2b", "qwen3-4b", "qwen3-8b",
               "internvl-2b", "internvl-4b", "internvl-8b"]

MODEL_LABELS = {
    "qwen3-2b":    "Qwen3-2B",
    "qwen3-4b":    "Qwen3-4B",
    "qwen3-8b":    "Qwen3-8B",
    "internvl-2b": "InternVL-2B",
    "internvl-4b": "InternVL-4B",
    "internvl-8b": "InternVL-8B",
}

MODALITY_SETTINGS  = ["zero_shot", "txt_only", "img_only", "full"]
COMPONENT_SETTINGS = ["full", "wo_subfig_type", "wo_local", "wo_global"]

SETTING_LABELS = {
    "zero_shot":      "Zero-shot",
    "txt_only":       "Text only",
    "img_only":       "Image only",
    "full":           "Full context",
    "wo_subfig_type": "w/o Subfig type",
    "wo_local":       "w/o Local text",
    "wo_global":      "w/o Global context",
}

QWEN_COLORS     = ["#4C72B0", "#55A868", "#C44E52"]
INTERNVL_COLORS = ["#8172B2", "#CCB974", "#64B5CD"]

mpl.rcParams.update({
    "font.family":    "Times New Roman",
    "font.size":      12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize":10,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
})


def compute_matcha_accuracy(matcha_dir: Path) -> pd.DataFrame:
    rows = []
    for fpath in matcha_dir.glob("*.csv"):
        m = re.match(r"(.+)_(zero_shot|this_work)\.csv$", fpath.name)
        if not m:
            continue
        model, setting = m.group(1), m.group(2)
        df = pd.read_csv(fpath)
        if "pred" not in df.columns or "gt_answer" not in df.columns:
            continue
        df = df[df["error"].isna() | (df["error"] == "")]
        df["pred_clean"] = df["pred"].astype(str).str.strip().str.upper().str[:1]
        df["gt_clean"]   = df["gt_answer"].astype(str).str.strip().str.upper().str[:1]
        acc = (df["pred_clean"] == df["gt_clean"]).mean()
        rows.append({"model": model, "setting": setting, "accuracy": round(acc, 4), "n": len(df)})
    return pd.DataFrame(rows)


def plot_matcha(df_acc: pd.DataFrame, out_dir: Path):
    pivot = df_acc.pivot(index="model", columns="setting", values="accuracy").reindex(MODEL_ORDER)
    pivot.index = [MODEL_LABELS[m] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(pivot))
    width = 0.35
    settings = [s for s in ["zero_shot", "this_work"] if s in pivot.columns]
    colors = ["#4C72B0", "#C44E52"]
    labels = ["Zero-shot", "This work"]

    for i, (s, c, l) in enumerate(zip(settings, colors, labels)):
        bars = ax.bar(x + (i - 0.5) * width, pivot[s], width, label=l,
                      color=c, alpha=0.85, edgecolor="white")
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("MatCha Benchmark Accuracy")
    ax.legend()
    ax.set_ylim(0, max(df_acc["accuracy"]) * 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(out_dir / "matcha_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(pivot.to_string())


def plot_cosine_heatmap(df: pd.DataFrame, settings: list, title: str, fname: str, out_dir: Path):
    top1 = df[df["rank"] == 1]
    pivot = (top1.groupby(["model", "setting"])["cosine"].mean()
             .unstack("setting").reindex(MODEL_ORDER).reindex(columns=settings))
    pivot.index = [MODEL_LABELS[m] for m in pivot.index]
    pivot.columns = [SETTING_LABELS[s] for s in settings]

    fig, ax = plt.subplots(figsize=(len(settings) * 1.8 + 1, len(MODEL_ORDER) * 0.7 + 1))
    im = ax.imshow(pivot.values.astype(float), cmap="YlOrRd",
                   vmin=pivot.values.min() - 0.01, vmax=pivot.values.max() + 0.01)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_lines(df, settings, title_suffix, fname, out_dir):
    top1 = df[df["rank"] == 1]
    pivot = (top1.groupby(["model", "setting"])["cosine"].mean()
             .unstack("setting").reindex(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, family, models, colors in zip(
        axes, ["Qwen3-VL", "InternVL-3.5"],
        [["qwen3-2b", "qwen3-4b", "qwen3-8b"], ["internvl-2b", "internvl-4b", "internvl-8b"]],
        [QWEN_COLORS, INTERNVL_COLORS],
    ):
        x = np.arange(len(settings))
        for model, color in zip(models, colors):
            if model not in pivot.index:
                continue
            vals = [pivot.loc[model, s] if s in pivot.columns else np.nan for s in settings]
            ax.plot(x, vals, marker="o", label=MODEL_LABELS[model], color=color, linewidth=1.8, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([SETTING_LABELS[s] for s in settings], rotation=15, ha="right")
        ax.set_ylabel("Mean Cosine Similarity (Top-1)")
        ax.set_title(family)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.suptitle(title_suffix, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    cfg = load_config("configs/config.yaml")
    cosine_merged = Path(cfg["COSINE_OUTPUT_DIR"]) / "all_cosine_merged.csv"
    matcha_dir    = Path(cfg["MATCHA_OUTPUT_DIR"])
    out_dir       = Path(cfg["ANALYSIS_OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cos = pd.read_csv(cosine_merged)
    df_acc = compute_matcha_accuracy(matcha_dir)

    if len(df_acc) > 0:
        plot_matcha(df_acc, out_dir)
        df_acc.to_csv(out_dir / "matcha_accuracy.csv", index=False)

    all_settings = MODALITY_SETTINGS + [s for s in COMPONENT_SETTINGS if s != "full"]
    plot_cosine_heatmap(df_cos, all_settings,
                        "Mean Cosine Similarity (Top-1) by Model and Setting",
                        "cosine_heatmap_all.png", out_dir)
    _plot_lines(df_cos, MODALITY_SETTINGS,
                "Modality Ablation: Effect of Input Modality", "modality_ablation.png", out_dir)
    _plot_lines(df_cos, COMPONENT_SETTINGS,
                "Component Ablation: Contribution of Each Evidence Component",
                "component_ablation.png", out_dir)

    top1 = df_cos[df_cos["rank"] == 1]
    summary = top1.groupby(["model", "setting"])["cosine"].agg(["mean", "std", "count"]).round(4)
    summary.to_csv(out_dir / "summary_table.csv")
    print(summary.to_string())
    print(f"\n[DONE] All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
