
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history_dict):
    plt.figure(figsize=(7, 4))
    for name, hist in history_dict.items():
        plt.plot(hist.train_loss, label=f"{name} train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    for name, hist in history_dict.items():
        plt.plot(hist.val_acc, label=f"{name} val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def summarize_results(results):
    df = pd.DataFrame(results).sort_values("val_acc", ascending=False).reset_index(drop=True)
    return df
