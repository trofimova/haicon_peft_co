"""
Create plots from saved Part 3 result files.

Usage:
    python notebooks/plot_part3_results.py outputs/part3/20260518-143000_eurosat

If no run directory is provided, the latest directory under outputs/part3 is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part3"
LINESTYLES = ["-", "--", ":", "-."]


def latest_run_dir(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No result directories found under {output_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    results_path = run_dir / "results.csv"
    history_path = run_dir / "history.csv"
    config_path = run_dir / "config.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    df = pd.read_csv(results_path)
    history_df = pd.read_csv(history_path) if history_path.exists() else None
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return df, history_df, config


def format_shot_axis(ax, shots) -> None:
    ticks = sorted({int(shot) for shot in shots})
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value)}"))
    ax.minorticks_off()
    ax.set_xlabel("Shots per class")


def plot_accuracy_grid(df: pd.DataFrame, config: dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_cols = ["source_acc"]
    plot_df = aggregate_results(df, metric_cols)

    for method, group in plot_df.groupby("method"):
        group = group.sort_values("shots_per_class")
        yerr = group["source_acc_std"] if "source_acc_std" in group.columns else None
        ax.errorbar(
            group["shots_per_class"],
            group["source_acc"],
            yerr=yerr,
            marker="o",
            capsize=3 if yerr is not None else 0,
            label=method,
        )

    format_shot_axis(ax, plot_df["shots_per_class"])
    ax.set_title("In-domain accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_shifted_accuracy(df: pd.DataFrame, config: dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_cols = ["target_acc"]
    plot_df = aggregate_results(df, metric_cols)

    for method, group in plot_df.groupby("method"):
        group = group.sort_values("shots_per_class")
        yerr = group["target_acc_std"] if "target_acc_std" in group.columns else None
        ax.errorbar(
            group["shots_per_class"],
            group["target_acc"],
            yerr=yerr,
            marker="o",
            capsize=3 if yerr is not None else 0,
            label=method,
        )

    shift = config.get("domain_shift_strength", "?")
    format_shot_axis(ax, plot_df["shots_per_class"])
    ax.set_title(f"Shifted-domain accuracy\n(domain shift = {shift})")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_budget_and_shift(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_df = df
    if "repeat_seed" in df.columns:
        metric_cols = [
            "trainable_params",
            "target_acc",
            "feature_shift",
        ]
        plot_df = df.groupby(["method", "shots_per_class"], as_index=False)[metric_cols].mean()

    scatter = axes[0].scatter(
        plot_df["trainable_params"],
        plot_df["target_acc"],
        c=plot_df["shots_per_class"],
        s=90,
        cmap="viridis",
        zorder=3,
    )
    for _, row in plot_df.iterrows():
        axes[0].annotate(
            row["method"],
            (row["trainable_params"], row["target_acc"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable parameters")
    axes[0].set_ylabel("Target accuracy")
    axes[0].set_title("Quality vs parameter budget")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=axes[0], label="shots/class")

    axes[1].scatter(plot_df["feature_shift"], plot_df["target_acc"], s=90, zorder=3)
    for _, row in plot_df.iterrows():
        label = f'{row["method"]} ({row["shots_per_class"]})'
        axes[1].annotate(
            label,
            (row["feature_shift"], row["target_acc"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    axes[1].set_xlabel("Mean feature cosine distance from frozen backbone")
    axes[1].set_ylabel("Target accuracy")
    axes[1].set_title("Did moving features help?")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def method_colors(methods):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {method: color_cycle[i % len(color_cycle)] for i, method in enumerate(methods)}


def shot_linestyles(shots):
    ordered = sorted(shots, reverse=True)
    return {shot: LINESTYLES[i % len(LINESTYLES)] for i, shot in enumerate(ordered)}


def aggregate_results(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    present_metrics = [metric for metric in metrics if metric in df.columns]
    group_cols = ["method", "shots_per_class"]
    if "repeat_seed" not in df.columns or df["repeat_seed"].nunique() <= 1:
        return df[group_cols + present_metrics].copy()

    mean_df = df.groupby(group_cols, as_index=False)[present_metrics].mean()
    std_df = df.groupby(group_cols, as_index=False)[present_metrics].std()
    std_df = std_df.rename(columns={metric: f"{metric}_std" for metric in present_metrics})
    return mean_df.merge(std_df, on=group_cols, how="left")


def plot_metric_lines(ax, df: pd.DataFrame, metric: str, colors: dict, ylabel: str, title: str):
    for method, group in df.groupby("method"):
        group = group.sort_values("shots_per_class")
        yerr = group[f"{metric}_std"] if f"{metric}_std" in group.columns else None
        ax.errorbar(
            group["shots_per_class"],
            group[metric],
            yerr=yerr,
            marker="o",
            capsize=3 if yerr is not None else 0,
            color=colors[method],
            label=method,
        )
    format_shot_axis(ax, df["shots_per_class"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_generalization_summary(df: pd.DataFrame):
    required = {
        "train_acc",
        "source_acc",
        "target_acc",
        "generalization_gap",
        "domain_gap",
        "train_loss",
        "source_loss",
        "target_loss",
    }
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Cannot plot generalization summary; missing columns: {missing}")

    metrics = sorted(required)
    plot_df = aggregate_results(df, metrics)
    methods = sorted(plot_df["method"].unique())
    colors = method_colors(methods)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for method, group in plot_df.groupby("method"):
        group = group.sort_values("shots_per_class")
        color = colors[method]
        for metric, linestyle, label in [
            ("train_acc", "-", "train"),
            ("source_acc", "--", "source val"),
            ("target_acc", ":", "target val"),
        ]:
            yerr = group[f"{metric}_std"] if f"{metric}_std" in group.columns else None
            axes[0, 0].errorbar(
                group["shots_per_class"],
                group[metric],
                yerr=yerr,
                marker="o",
                linestyle=linestyle,
                capsize=3 if yerr is not None else 0,
                color=color,
                label=f"{method} {label}",
            )

        for metric, linestyle, label in [
            ("train_loss", "-", "train"),
            ("source_loss", "--", "source val"),
            ("target_loss", ":", "target val"),
        ]:
            yerr = group[f"{metric}_std"] if f"{metric}_std" in group.columns else None
            axes[1, 0].errorbar(
                group["shots_per_class"],
                group[metric],
                yerr=yerr,
                marker="o",
                linestyle=linestyle,
                capsize=3 if yerr is not None else 0,
                color=color,
                label=f"{method} {label}",
            )

    axes[0, 0].set_title("Accuracy: train vs source vs target")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Loss: train vs source vs target")
    axes[1, 0].set_ylabel("Loss")

    plot_metric_lines(
        axes[0, 1], plot_df, "generalization_gap", colors,
        "Train acc - source acc", "Generalization gap",
    )
    plot_metric_lines(
        axes[0, 2], plot_df, "domain_gap", colors,
        "Source acc - target acc", "Domain gap",
    )
    plot_metric_lines(
        axes[1, 1], plot_df, "source_loss", colors,
        "Source validation loss", "Source validation loss",
    )
    plot_metric_lines(
        axes[1, 2], plot_df, "target_loss", colors,
        "Target validation loss", "Target validation loss",
    )

    for ax in axes.ravel():
        format_shot_axis(ax, plot_df["shots_per_class"])
        ax.grid(True, alpha=0.3)

    method_handles = [
        Line2D([0], [0], color=colors[method], lw=2, label=method)
        for method in methods
    ]
    split_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="train"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="source val"),
        Line2D([0], [0], color="black", lw=2, linestyle=":", label="target val"),
    ]
    method_legend = axes[0, 0].legend(
        handles=method_handles, title="Method", fontsize=8, title_fontsize=9,
        loc="lower right", handlelength=3.2,
    )
    axes[0, 0].add_artist(method_legend)
    axes[1, 0].legend(
        handles=split_handles, title="Split", fontsize=8, title_fontsize=9,
        loc="upper right", handlelength=4.0,
    )

    fig.tight_layout()
    return fig


def plot_training_curves(history_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    methods = sorted(history_df["method"].unique())
    shots_values = sorted(history_df["shots_per_class"].unique(), reverse=True)
    colors = method_colors(methods)
    linestyles = shot_linestyles(shots_values)
    has_train_acc = "train_acc" in history_df.columns

    group_cols = ["method", "shots_per_class", "epoch"]
    mean_cols = ["train_loss", "val_loss", "val_acc"]
    if has_train_acc:
        mean_cols.append("train_acc")
    mean_df = history_df.groupby(group_cols, as_index=False)[mean_cols].mean()

    repeat_groups = []
    if "repeat_seed" in history_df.columns and history_df["repeat_seed"].nunique() > 1:
        repeat_groups = list(history_df.groupby(["method", "shots_per_class", "repeat_seed"]))

    for (method, shots, _repeat_seed), group in repeat_groups:
        group = group.sort_values("epoch")
        style = {
            "color": colors[method],
            "linestyle": linestyles[shots],
            "linewidth": 0.8,
            "alpha": 0.18,
        }
        axes[0, 0].plot(group["epoch"], group["train_loss"], **style)
        axes[0, 1].plot(group["epoch"], group["val_loss"], **style)
        if has_train_acc:
            axes[1, 0].plot(group["epoch"], group["train_acc"], **style)
        axes[1, 1].plot(group["epoch"], group["val_acc"], **style)

    for (method, shots), group in mean_df.groupby(["method", "shots_per_class"]):
        group = group.sort_values("epoch")
        style = {
            "color": colors[method],
            "linestyle": linestyles[shots],
            "linewidth": 2.0,
        }
        axes[0, 0].plot(group["epoch"], group["train_loss"], **style)
        axes[0, 1].plot(group["epoch"], group["val_loss"], **style)
        if has_train_acc:
            axes[1, 0].plot(group["epoch"], group["train_acc"], **style)
        axes[1, 1].plot(group["epoch"], group["val_acc"], **style)

    axes[0, 0].set_ylabel("Train loss")
    axes[0, 0].set_title("Training loss")

    axes[0, 1].set_ylabel("Source validation loss")
    axes[0, 1].set_title("Validation loss")

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Train accuracy")
    axes[1, 0].set_title("Training accuracy")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Source validation accuracy")
    axes[1, 1].set_title("Validation accuracy")
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    method_handles = [
        Line2D([0], [0], color=colors[method], lw=2, label=method)
        for method in methods
    ]
    shot_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=linestyles[shots],
               label=f"{shots} shots")
        for shots in shots_values
    ]
    method_legend = axes[1, 1].legend(
        handles=method_handles, title="Method", fontsize=8, title_fontsize=9,
        loc="lower right", handlelength=3.2,
    )
    axes[1, 1].add_artist(method_legend)
    axes[0, 0].legend(
        handles=shot_handles, title="Shots/class", fontsize=8, title_fontsize=9,
        loc="upper right", handlelength=4.0,
    )

    fig.tight_layout()
    return fig


def main():
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_run_dir()
    df, history_df, config = load_run(run_dir)

    fig = plot_accuracy_grid(df, config)
    fig.savefig(run_dir / "accuracy_grid.png", dpi=160)
    plt.close(fig)

    fig = plot_shifted_accuracy(df, config)
    fig.savefig(run_dir / "shifted_accuracy.png", dpi=160)
    plt.close(fig)

    fig = plot_generalization_summary(df)
    fig.savefig(run_dir / "generalization_summary.png", dpi=160)
    plt.close(fig)

    if history_df is not None and not history_df.empty:
        fig = plot_training_curves(history_df)
        fig.savefig(run_dir / "training_curves.png", dpi=160)
        plt.close(fig)

    print(f"Saved plots to {run_dir}")


if __name__ == "__main__":
    main()
