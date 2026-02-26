"""YOLO training loss visualisation.

Provides :class:`YOLOTrainingVisualizer` for plotting box/cls/dfl losses
from Ultralytics ``results.csv`` files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class YOLOTrainingVisualizer:
    """Visualise YOLO training metrics from a ``results.csv`` file.

    Args:
        results_path: Path to the Ultralytics ``results.csv``.
        figsize: Default figure size ``(width, height)`` in inches.
    """

    def __init__(
        self,
        results_path: str,
        figsize: tuple = (18, 5),
    ) -> None:
        self.results_path = Path(results_path)
        self.figsize = figsize
        self.df: Optional[pd.DataFrame] = None
        self.loss_metrics = {
            "box": {"train": "train/box_loss", "val": "val/box_loss", "title": "Box Loss"},
            "cls": {"train": "train/cls_loss", "val": "val/cls_loss", "title": "Class Loss"},
            "dfl": {"train": "train/dfl_loss", "val": "val/dfl_loss", "title": "DFL Loss"},
        }

    # ── Data loading ────────────────────────────────────────────────────

    def load_data(self) -> bool:
        """Load and prepare the CSV data."""
        try:
            self.df = pd.read_csv(self.results_path)
            self.df.columns = self.df.columns.str.strip()
            return True
        except FileNotFoundError:
            print(f"Error: {self.results_path} not found.")
            return False
        except Exception as exc:
            print(f"Error loading data: {exc}")
            return False

    def validate_columns(self) -> bool:
        """Check that required columns exist in the loaded dataframe."""
        missing = [
            col
            for m in self.loss_metrics.values()
            for col in m.values()
            if col not in self.df.columns and col != m.get("title")
        ]
        if missing:
            print(f"Warning: Missing columns: {missing}")
            print(f"Available: {self.df.columns.tolist()}")
            return False
        return True

    # ── Plotting ────────────────────────────────────────────────────────

    def plot_metric(self, ax: plt.Axes, metric_key: str, add_stats: bool = True) -> None:
        """Plot a single loss metric (train + val) on *ax*."""
        m = self.loss_metrics[metric_key]
        ax.plot(self.df["epoch"], self.df[m["train"]],
                label=f"Training {m['title']}", linewidth=2)
        ax.plot(self.df["epoch"], self.df[m["val"]],
                label=f"Validation {m['title']}", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(m["title"], fontsize=12)
        ax.set_title(f"{m['title']} vs. Epochs", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        if add_stats:
            self._add_statistics(ax, m)

    def _add_statistics(self, ax: plt.Axes, metric: Dict[str, str]) -> None:
        """Annotate minimum validation loss on the plot."""
        val_data = self.df[metric["val"]]
        min_val = val_data.min()
        min_epoch = self.df.loc[val_data.idxmin(), "epoch"]
        ax.annotate(
            f"Min Val: {min_val:.4f}\n@ Epoch {min_epoch}",
            xy=(min_epoch, min_val),
            xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
        )

    def plot_all_losses(self, add_stats: bool = True) -> None:
        """Create subplots for box, class, and DFL losses."""
        if not self.load_data() or not self.validate_columns():
            return

        fig, axs = plt.subplots(1, 3, figsize=self.figsize)
        for idx, key in enumerate(["box", "cls", "dfl"]):
            self.plot_metric(axs[idx], key, add_stats)

        plt.tight_layout()
        plt.show()
        self.plot_total_loss()

    def plot_total_loss(self) -> None:
        """Plot total training/validation loss if available."""
        if self.df is None or "train/loss" not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.df["epoch"], self.df["train/loss"],
                label="Total Training Loss", linewidth=2)
        ax.plot(self.df["epoch"], self.df["val/loss"],
                label="Total Validation Loss", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Total Loss", fontsize=12)
        ax.set_title("Total Loss vs. Epochs", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        min_val = self.df["val/loss"].min()
        min_epoch = self.df.loc[self.df["val/loss"].idxmin(), "epoch"]
        ax.annotate(
            f"Min: {min_val:.4f} @ Epoch {min_epoch}",
            xy=(min_epoch, min_val),
            xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
        )
        plt.show()

    # ── Save / summarise ────────────────────────────────────────────────

    def save_plots(self, output_dir: str = "plots", dpi: int = 300) -> None:
        """Save individual loss plots to files."""
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        if not self.load_data() or not self.validate_columns():
            return

        for key in ["box", "cls", "dfl"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            self.plot_metric(ax, key)
            plt.tight_layout()
            plt.savefig(out / f"{key}_loss.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        print(f"Plots saved to {out}")

    def get_training_summary(self) -> Optional[Dict[str, Any]]:
        """Return a dict summarising training results."""
        if not self.load_data():
            return None

        summary: Dict[str, Any] = {
            "total_epochs": len(self.df),
            "final_epoch": int(self.df["epoch"].iloc[-1]),
        }
        for name, cols in self.loss_metrics.items():
            if cols["val"] in self.df.columns:
                vals = self.df[cols["val"]]
                summary[f"{name}_min_val_loss"] = float(vals.min())
                summary[f"{name}_min_val_epoch"] = int(
                    self.df.loc[vals.idxmin(), "epoch"]
                )
                summary[f"{name}_final_val_loss"] = float(vals.iloc[-1])
        return summary
