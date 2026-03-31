from src.utils.config import load_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_keypoint_errors(df):
    """Vrátí dict: label -> pole chyb"""
    errors = {}

    for col in df.columns:
        if col.endswith("_ann_x"):
            label = col.replace("_ann_x", "")

            x_ann = df[f"{label}_ann_x"]
            y_ann = df[f"{label}_ann_y"]
            x_pred = df[f"{label}_pred_x"]
            y_pred = df[f"{label}_pred_y"]

            err = np.sqrt((x_pred - x_ann)**2 + (y_pred - y_ann)**2)
            errors[label] = err

    return errors


def analysis_per_keypoint(errors, output_dir):
    print("\n=== Per-keypoint analysis ===")

    stats = {}

    for label, err in errors.items():
        stats[label] = {
            "mean": np.mean(err),
            "median": np.median(err),
            "std": np.std(err),
            "max": np.max(err)
        }
        print(f"{label}: "
              f"mean={stats[label]['mean']:.2f}, "
              f"median={stats[label]['median']:.2f}, "
              f"std={stats[label]['std']:.2f}, "
              f"max={stats[label]['max']:.2f}")

    keypoint_df = pd.DataFrame(stats).T.reset_index()
    keypoint_df = keypoint_df.rename(columns={"index": "keypoint"})
    keypoint_df.to_csv(os.path.join(output_dir, "per_keypoint_stats.csv"), index=False)

    return stats


def analysis_per_image(df, errors, output_dir):
    print("\n=== Per-image analysis ===")

    err_df = pd.DataFrame(errors)

    df["mean_error"] = err_df.mean(axis=1)
    df["max_error"] = err_df.max(axis=1)

    print("\nTop 5 nejhorších obrázků:")
    print(df.sort_values("mean_error", ascending=False)[
        ["filename", "mean_error", "max_error"]
    ].head())

    df_results = df[["filename", "mean_error", "max_error"]]
    df_results.to_csv(os.path.join(output_dir, "per_image_stats.csv"), index=False)

    return df_results


def plot_distributions(errors, output_dir):
    print("\n=== Plotting distributions ===")

    os.makedirs(output_dir, exist_ok=True)

    # === Histogram ===
    plt.figure(figsize=(10, 6))

    all_errors = np.concatenate(list(errors.values()))
    plt.hist(all_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

    plt.title("Error Distribution (All Keypoints)")
    plt.xlabel("Pixel Error")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram.png"))
    plt.close()

    # === Boxplot ===
    plt.figure(figsize=(12, 6))

    labels = list(errors.keys())
    data = [errors[label] for label in labels]

    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=45)

    plt.title("Error per Keypoint")
    plt.ylabel("Pixel Error")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot.png"))
    plt.close()


def plot_keypoint_distributions(errors, output_dir):
    """Vytvoří histogramy chyb pro každý keypoint zvlášť"""
    print("\n=== Plotting per-keypoint distributions ===")
    os.makedirs(output_dir, exist_ok=True)

    for label, err in errors.items():
        plt.figure(figsize=(10, 6))
        plt.hist(err, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f"Error Distribution for Keypoint: {label}")
        plt.xlabel("Pixel Error")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"histogram_{label}.png"))
        plt.close()


def plot_keypoint_stats_boxplot(stats, output_dir):
    print("\n=== Plotting keypoint stats boxplot ===")
    os.makedirs(output_dir, exist_ok=True)

    stats_df = pd.DataFrame(stats).T

    plt.figure(figsize=(10, 6))

    data = [stats_df[col] for col in ["mean", "median", "std", "max"]]
    plt.boxplot(data, tick_labels=["Mean", "Median", "Std", "Max"], showfliers=False)

    plt.title("Summary of Keypoint Errors Across All Keypoints")
    plt.ylabel("Pixel Error")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "keypoint_stats_boxplot.png"))
    plt.close()


def plot_metric_distributions(stats, output_dir):
    print("\n=== Plotting metric distributions per keypoint ===")
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["mean", "median", "std", "max"]

    stats_df = pd.DataFrame(stats).T

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        data = stats_df[metric]
        plt.boxplot(data, tick_labels=[metric], showfliers=False)
        plt.title(f"Distribution of {metric} error across keypoints")
        plt.ylabel("Pixel Error")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"boxplot_{metric}.png"))
        plt.close()


def plot_metrics_per_keypoint(stats, output_dir):
    print("\n=== Plotting metrics per keypoint ===")
    os.makedirs(output_dir, exist_ok=True)

    stats_df = pd.DataFrame(stats).T
    stats_df = stats_df.reset_index().rename(columns={"index": "keypoint"})

    metrics = ["mean", "median", "std", "max"]

    for metric in metrics:
        plt.figure(figsize=(14, 6))
        plt.bar(stats_df["keypoint"], stats_df[metric],
                color='skyblue', edgecolor='black')
        plt.xticks(rotation=45)
        plt.title(f"{metric.capitalize()} Error per Keypoint")
        plt.ylabel("Pixel Error")
        plt.xlabel("Keypoint")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_per_keypoint.png"))
        plt.close()


def main():
    # === Config ===
    cfg = load_config("configs/config_list.yaml")
    print("Loaded configuration:", cfg)
    exp = cfg["experiment"]["name"]
    prefix = cfg["predict"]["prefix"]
    csv_path = os.path.join("results", exp, "predictions_vs_annotations.csv")
    output_dir =os.path.join("results", exp, "analysis", prefix)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # === Compute errors ===
    errors = compute_keypoint_errors(df)

    # === Per keypoint ===
    stats = analysis_per_keypoint(errors, output_dir)

    # === Per image ===
    df_results = analysis_per_image(df, errors, output_dir)

    # === Distribuce ===
    plot_distributions(errors, output_dir)

    # === Distribuce pro každý keypoint ===
    plot_keypoint_distributions(errors, output_dir)

    # === Boxplot pro stats ===
    plot_keypoint_stats_boxplot(stats, output_dir)

    # === Grafy metrik ===
    plot_metrics_per_keypoint(stats, output_dir)

    print("\nHotovo. Výstupy jsou v:", output_dir)


if __name__ == "__main__":
    main()