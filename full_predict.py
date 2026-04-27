import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
import csv
import shutil
import json
import seaborn as sns

from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.metrics.factory import get_metrics
from src.train.factory import get_trainer

def batch_loader(dataset, batch_size=1):
   """Generator to load dataset in batches, stacking NumPy arrays.

   Args:
      dataset (list or Dataset): Dataset with __getitem__ method.
      batch_size (int, optional): Number of samples per batch. Defaults to 1.

   Yields:
      dict: Batch dictionary with stacked arrays or list of values.
   """
   n = len(dataset)
   for i in range(0, n, batch_size):
      batch_samples = [dataset[j] for j in range(i, min(i + batch_size, n))]

      batch = {}
      for key in batch_samples[0]:
         values = [s[key] for s in batch_samples]
         if isinstance(values[0], np.ndarray):
               batch[key] = np.stack(values)
         else:
               batch[key] = values  # metadata, filename, etc.

      yield batch

def visualize_heatmaps_combined(image, heatmaps, path_to_save, fname="heatmaps_combined.png",
                                show_points=True, threshold=0.05, keypoints=None):
   """Visualize combined heatmaps overlaid on an image.

   Args:
      image (np.ndarray or torch.Tensor): Input image [C,H,W] or [H,W,3].
      heatmaps (np.ndarray or torch.Tensor): Heatmaps [N,64,64].
      path_to_save (str): Directory to save figure.
      fname (str, optional): Filename for saved figure. Defaults to "heatmaps_combined.png".
      show_points (bool, optional): Whether to show maxima and keypoints. Defaults to True.
      threshold (float, optional): Minimum heatmap max to include. Defaults to 0.05.
      keypoints (list of np.ndarray, optional): Keypoints [num_points, 2]. Defaults to None.
   """
   # Convert torch tensors to numpy
   if torch.is_tensor(image):
      image = image.cpu().numpy()
   if torch.is_tensor(heatmaps):
      heatmaps = heatmaps.cpu().numpy()

   # Convert image to HWC if needed
   if image.ndim == 3 and image.shape[0] in (1, 3):
      image = image.transpose(1, 2, 0)
   image = np.clip(image, 0, 1)
   H, W = image.shape[:2]

   # Combine heatmaps
   combined = np.zeros((H, W), dtype=np.float32)
   for hm in heatmaps:
      hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
      hm_resized = np.clip(hm_resized, 0, None)
      if hm_resized.max() < threshold:
         continue
      if hm_resized.max() > 0:
         hm_resized /= hm_resized.max()
      combined += hm_resized

   if combined.max() > 0:
      combined /= combined.max()

   fig, ax = plt.subplots(figsize=(6, 6))
   ax.imshow(image)
   ax.imshow(combined, cmap="turbo", alpha=0.4)

   if show_points:
      # Mark maxima of each heatmap
      for hm in heatmaps:
         hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
         y, x = np.unravel_index(np.argmax(hm_resized), hm_resized.shape)
         ax.scatter(x, y, c="red", s=6, label="Prediction")

      # Overlay keypoints if provided
      if keypoints is not None:
         for i, kp_set in enumerate(keypoints):
               kp_set = np.atleast_2d(kp_set).copy()
               kp_set[:, 0] = kp_set[:, 0] * W / 64
               kp_set[:, 1] = kp_set[:, 1] * H / 64
               ax.scatter(kp_set[:, 0], kp_set[:, 1], c="green", s=6, marker="x",
                        label="Anotation" if i == 0 else None)

   handles, labels = ax.get_legend_handles_labels()
   by_label = dict(zip(labels, handles))
   ax.legend(by_label.values(), by_label.keys())

   ax.axis("off")
   os.makedirs(path_to_save, exist_ok=True)
   plt.savefig(os.path.join(path_to_save, fname), bbox_inches="tight")
   plt.close(fig)

def predict_images(cfg, trainer, loader, metrics, folder_path):
   path_to_save = os.path.join(folder_path, "prediction")
   os.makedirs(path_to_save, exist_ok=True)
   results = []

   # === Prediction loop ===
   for batch, preds in trainer.predict_image(loader, checkpoint=cfg["predict"]["model"]):
      for img, pred, fname, keypoints in zip(
         batch["image"], preds, batch["filename"], batch["keypoints"]
      ):
         # Visualize combined heatmap
         visualize_heatmaps_combined(img, pred, path_to_save, fname,
                                       show_points=True, keypoints=keypoints)

         # Compute metrics
         metrics_values = {}
         for m in metrics:
               m.reset()
               m.update(np.expand_dims(pred, axis=0), np.expand_dims(keypoints, axis=0))
               metrics_values[type(m).__name__] = m.compute()

         # Add filename to results
         row = {"filename": fname}
         row.update(metrics_values)
         results.append(row)

   # Sort results by metric
   metrics = list(metrics_values.keys())

   results_sorted = sorted(
      results,
      key=lambda x: (x[metrics[0]], -x[metrics[1]]),
      reverse=True
   )

   # Save results to CSV
   csv_path = os.path.join(folder_path, f"{cfg["predict"]["prefix"]}_metrics_sorted.csv")
   with open(csv_path, "w", newline="") as f:
      writer = csv.writer(f)
      header = ["filename"] + list(metrics_values.keys())
      writer.writerow(header)
      for row in results_sorted:
         writer.writerow([row[h] for h in header])

   print(f"Results saved to {csv_path}")

def copy_files(rows, base_dir, folder_path, image_column):
   os.makedirs(base_dir, exist_ok=True)

   for row in rows:
      filename = row[image_column]

      src = os.path.join(folder_path, "prediction", filename)

      if not os.path.exists(src):
         print(f"Soubor neexistuje: {src}")
         continue

      filename = os.path.basename(src)
      dst = os.path.join(base_dir, filename)

      # copy originálu
      shutil.copy2(src, dst)

def select_images(cfg, folder_path):
   csv_path = os.path.join(folder_path, f"{cfg["predict"]["prefix"]}_metrics_sorted.csv")
   image_column = "filename"  # název sloupce v CSV
   x = 10  # kolik řádků vzít

   # výstupní složky
   best_base = os.path.join(folder_path, "prediction_selection/best", cfg["predict"]["prefix"])
   worst_base = os.path.join(folder_path, "prediction_selection/worst", cfg["predict"]["prefix"])

   # ==== NAČTENÍ CSV ====
   rows = []

   with open(csv_path, newline='', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
         rows.append(row)

   # první a poslední x
   best_rows = rows[:cfg["predict"]["num_select_images"]]
   worst_rows = rows[-cfg["predict"]["num_select_images"]:]

   copy_files(best_rows, best_base, folder_path, image_column)
   copy_files(worst_rows, worst_base, folder_path, image_column)

def make_dataset_of_prediction(cfg, trainer, dataset, annotations_label):
   records = []

   for item, heatmaps_np in trainer.predict(dataset, checkpoint=cfg["predict"]["model"], batch_size=cfg["predict"]["batch_size"]):
      preds_np = heatmaps_np
      filenames = item["filename"]
      annotations = item["keypoints"]

      for batch_idx in range(preds_np.shape[0]):
         row = {"filename": filenames[batch_idx]}
         for keypoint_idx in range(preds_np.shape[1]):
               heatmap = preds_np[batch_idx, keypoint_idx]
               y_pred, x_pred = np.unravel_index(np.argmax(heatmap), heatmap.shape)
               x_ann, y_ann = annotations[batch_idx][keypoint_idx]
               label = annotations_label[keypoint_idx]
               row.update({
                  f"{label}_ann_x": int(x_ann),
                  f"{label}_ann_y": int(y_ann),
                  f"{label}_pred_x": int(x_pred),
                  f"{label}_pred_y": int(y_pred),
               })
         records.append(row)

   df = pd.DataFrame(records)
   return df

def save_dataframe_csv(cfg, folder_path, df):
    """Save DataFrame of predictions to CSV."""
    output_csv = os.path.join(folder_path, f"{cfg["predict"]["prefix"]}_predictions_vs_annotations.csv")
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to: {output_csv}")

def save_dataframe_json(cfg, folder_path, df):
    """Save DataFrame of predictions to JSON in keypoints-per-file format."""
    output_json = os.path.join(folder_path, f"{cfg["predict"]["prefix"]}_keypoints.json")

    results = []
    for _, row in df.iterrows():
        keypoints = {}
        for col in df.columns:
            if "_pred_x" in col:
                label = col.replace("_pred_x", "")
                keypoints[label] = {
                    "x": int(row[f"{label}_pred_x"]),
                    "y": int(row[f"{label}_pred_y"])
                }
        results.append({
            "filename": row["filename"],
            "keypoints": keypoints
        })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"JSON saved to: {output_json}")

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

def analysis_per_keypoint(cfg, errors, output_dir):
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
   keypoint_df.to_csv(os.path.join(output_dir, f"{cfg["predict"]["prefix"]}_per_keypoint_stats.csv"), index=False)

   return stats

def analysis_per_image(cfg, df, errors, output_dir):
   print("\n=== Per-image analysis ===")

   err_df = pd.DataFrame(errors)

   df["mean_error"] = err_df.mean(axis=1)
   df["max_error"] = err_df.max(axis=1)

   print("\nTop 5 nejhorších obrázků:")
   print(df.sort_values("mean_error", ascending=False)[
      ["filename", "mean_error", "max_error"]
   ].head())

   df_results = df[["filename", "mean_error", "max_error"]]
   df_results.to_csv(os.path.join(output_dir, f"{cfg["predict"]["prefix"]}_per_image_stats.csv"), index=False)

   return df_results

def analysis_of_mean_error(df, stats):
   # vytáhni mean hodnoty
   means = [v['mean'] for v in stats.values()]

   # výpočty
   mean_of_means = np.mean(means)
   std_of_means = np.std(means, ddof=1)  # sample std

   print("mean of means:", mean_of_means)
   print("std of means:", std_of_means)
   return mean_of_means, std_of_means

def analysis_metrics(cfg, df):
   """
   Compute metrics (e.g. AED) from dataframe with keypoints.

   Args:
      df (pd.DataFrame): DataFrame with columns like:
         <kp>_ann_x, <kp>_ann_y, <kp>_pred_x, <kp>_pred_y
      metrics (list): list of metric instances (e.g. [AED(), ...])

   Returns:
      dict: {metric_name: value}
   """

   # --- 1. najdi keypointy ---
   keypoints = sorted({
      c.replace("_ann_x", "")
      for c in df.columns if "_ann_x" in c
   })

   B = len(df)
   K = len(keypoints)

   # --- 2. vytvoř pole ---
   preds = np.zeros((B, K, 2))
   targets = np.zeros((B, K, 2))

   for i, kp in enumerate(keypoints):
      targets[:, i, 0] = df[f"{kp}_ann_x"]
      targets[:, i, 1] = df[f"{kp}_ann_y"]

      preds[:, i, 0] = df[f"{kp}_pred_x"]
      preds[:, i, 1] = df[f"{kp}_pred_y"]

   # --- 3. spočítej metriky ---
   metrics = get_metrics(cfg["predict"]["kp_metrics"])
   metrics_values = {}

   for m in metrics:
      m.reset()
      m.update(preds, targets)
      metrics_values[type(m).__name__] = m.compute()

   print(metrics_values)

   return metrics_values

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

    # Seaborn theme + větší fonty
    sns.set_theme(style="whitegrid", context="talk")  
    # context="talk" zvětší texty (můžeš dát i "poster" pro ještě větší)

    stats_df = pd.DataFrame(stats).T
    stats_df = stats_df.reset_index().rename(columns={"index": "keypoint"})

    metrics = ["mean", "median", "std", "max"]

    for metric in metrics:
        plt.figure(figsize=(16, 7))

        ax = sns.barplot(
            data=stats_df,
            x="keypoint",
            y=metric,
            color="skyblue",
            edgecolor="black"
        )

        plt.xticks(rotation=45, ha="right")

        plt.title(f"{metric.capitalize()} Error per Keypoint", fontsize=20, weight="bold")
        plt.ylabel("Pixel Error", fontsize=16)
        plt.xlabel("Keypoint", fontsize=16)

        # zvětšení tick labelů
        ax.tick_params(axis='both', labelsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_per_keypoint.png"), dpi=300)
        plt.close()

def save_results(cfg, mean, std, metrics_dict, folder_path):
    """
    Uloží metriky do JSON souboru.

    Args:
        mean (float)
        std (float)
        metrics_dict (dict)
        filepath (str)
    """

    # převod numpy typů na klasické floaty
    metrics_clean = {
        k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
        for k, v in metrics_dict.items()
    }

    data = {
        "mean_of_means": float(mean),
        "std_of_means": float(std),
        "metrics": metrics_clean
    }

    path = os.path.join(folder_path, f"{cfg["predict"]["prefix"]}_results.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Uloženo do {path}")

def analyze_prediction(cfg, folder_path, df):
   output_dir = os.path.join(folder_path, "analysis", cfg["predict"]["prefix"])
   os.makedirs(output_dir, exist_ok=True)
   # === Compute errors ===
   errors = compute_keypoint_errors(df)

   # === Per keypoint ===
   stats = analysis_per_keypoint(cfg, errors, folder_path)

   # === Per image ===
   df_results = analysis_per_image(cfg, df, errors, folder_path)

   # === Of keypoints ===
   mean_of_means, std_of_means = analysis_of_mean_error(df, stats)

   # metriky spočítat
   metrics = analysis_metrics(cfg, df)

   # === Distribuce ===
   plot_distributions(errors, output_dir)

   # === Distribuce pro každý keypoint ===
   plot_keypoint_distributions(errors, output_dir)

   # === Boxplot pro stats ===
   plot_keypoint_stats_boxplot(stats, output_dir)

   # === Grafy metrik ===
   plot_metrics_per_keypoint(stats, output_dir)

   # full model results
   save_results(cfg, mean_of_means, std_of_means, metrics, folder_path)

   print("\nHotovo. Výstupy jsou v:", output_dir)

def main():
   # načti konfig a potřební věciinicializuj
   """Main function to load model, dataset, run predictions, visualize and save metrics."""
   # Load configuration
   cfg = load_config("configs/config_list.yaml")
   print("Configuration loaded.")
   print(cfg)

   # Create model
   model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
   print(f"Model '{cfg['model']['name']}' created.")

   # Load dataset
   dataset = get_dataset(
      name=cfg["predict"]["name"],
      load=cfg["predict"]["images_list"],
      num_samples=cfg["predict"]["num_samples"],
      keypoint_format=cfg["keypoint_format"],
      **cfg["predict"]["params"]
   )
   print(f"Dataset loaded: {len(dataset)} images.")

   # Load annotation labels from text file
   txt_path = os.path.join(cfg["predict"]["params"]["root_dir"],
                           cfg["predict"]["annotation_label"])
   annotations_label = []
   with open(txt_path, "r") as f:
      for line in f:
         line = line.strip()
         if line:
               annotations_label.append(line)
   print("Annotation labels:", annotations_label)

   # Wrap dataset in batch loader
   loader = batch_loader(dataset, batch_size=cfg["predict"]["batch_size"])

   # Metrics
   metrics = get_metrics(cfg["predict"]["metrics"])
   print(f"Metrics initialized: {cfg['predict']['metrics']}")

   # Trainer (for prediction logic)
   trainer = get_trainer(
      backend=cfg["model"]["backend"],
      model=model,
      train_dataset=None,
      val_dataset=None,
      loss_fn=None,
      metrics=None,
      experiment_name=cfg["experiment"]["name"],
      special_mode=cfg["model"].get("special_mode"),
      **cfg["train"]
   )
   print("Trainer initialized.")

   folder_path = os.path.join("results", cfg["experiment"]["name"])

   # # predikuj obrázky
   # predict_images(cfg, trainer, loader, metrics, folder_path)

   # # ulož best/worst
   # select_images(cfg, folder_path)

   # vytvoř dataset predikcí
   df = make_dataset_of_prediction(cfg, trainer, dataset, annotations_label)

   # ulož csv
   save_dataframe_csv(cfg, folder_path, df)

   # ulož json
   save_dataframe_json(cfg, folder_path, df)

   # analyzuj predikce
   analyze_prediction(cfg, folder_path, df)

if __name__ == "__main__":
    main()