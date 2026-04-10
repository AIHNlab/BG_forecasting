#!/usr/bin/env python
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmaps(filepath: str, savepath: str | None = None) -> None:
    # ---- Load and tidy data ----
    # Multi-row header: first row = tag, second row = metric (RMSE/MAE)
    df = pd.read_excel(filepath, sheet_name="stacked_by_tag", header=[0, 1])

    # Flatten multiindex columns: "<tag>_<metric>" except for dataset / horizon
    flat_cols = []
    for top, bottom in df.columns:
        if str(bottom).startswith("Unnamed"):
            flat_cols.append(top)
        else:
            flat_cols.append(f"{top}_{bottom}")
    df.columns = flat_cols

    # Fill down dataset labels
    df["dataset"] = df["dataset"].ffill()

    # ---- Label mappings ----
    horizons = [6, 12, 24]
    metrics = ["RMSE", "MAE"]

    horizon_labels = {6: "30min", 12: "60min", 24: "120min"}
    dataset_labels = {
        "0-0-0": "All Features",
        "0-0-1": "BG&CHO",
        "0-1-0": "BG&Bolus",
        "0-1-1": "BG",
    }
    tag_labels = {
        "2304": "8 days",
        "888": "3 days",
        "312": "1 day",
        "96": "6 hours",
    }

    # ---- Plot ----
    sns.set(style="white")

    fig, axes = plt.subplots(len(metrics), len(horizons), figsize=(18, 10))

    for i, metric in enumerate(metrics):
        for j, h in enumerate(horizons):
            # Filter one horizon
            subset = df[df["horizon"] == h].drop(columns=["horizon"])

            # Select columns for this metric (e.g. "*_RMSE")
            metric_cols = [c for c in subset.columns if c.endswith(metric)]

            # Build matrix: index = dataset, columns = tag
            heatmap_data = subset.set_index("dataset")[metric_cols]

            # Relabel columns from raw tag numbers to "8 days", "3 days", etc.
            new_cols = []
            for c in heatmap_data.columns:
                base_tag = c.replace(f"_{metric}", "")
                new_cols.append(tag_labels.get(base_tag, base_tag))
            heatmap_data.columns = new_cols

            # Relabel row index for datasets
            heatmap_data.index = [
                dataset_labels.get(idx, idx) for idx in heatmap_data.index
            ]

            ax = axes[i, j]
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".3f",
                cmap="coolwarm_r",
                ax=ax,
                cbar=True,  # individual colorbar per subplot
            )

            ax.set_title(f"{metric} ({horizon_labels[h]})")
            ax.set_ylabel("Included Features")
            ax.set_xlabel("Historic Context")

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create RMSE/MAE heatmaps from aggregated_tidy_rmse_mae_stacked_by_tag.xlsx"
    )
    parser.add_argument(
        "filepath",
        help="Path to aggregated_tidy_rmse_mae_stacked_by_tag.xlsx",
    )
    parser.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Optional path to save the figure (e.g. heatmaps.png)",
    )
    args = parser.parse_args()

    plot_heatmaps(args.filepath, savepath=args.out)


if __name__ == "__main__":
    main()
