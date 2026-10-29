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
from src.datasets.transforms.factory import get_transform
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
                batch[key] = values

        yield batch


def visualize_heatmaps_combined(image, heatmaps, path_to_save, fname="heatmaps_combined.png",
                                show_points=True, threshold=0.05, keypoints=None, output_shape=None):
    """Visualize combined heatmaps overlaid on an image.

    Args:
        image (np.ndarray or torch.Tensor): Input image [C,H,W] or [H,W,3].
        heatmaps (np.ndarray or torch.Tensor): Predicted heatmaps [N,H,W].
        path_to_save (str): Directory to save the resulting figure.
        fname (str, optional): Filename for the saved figure. Defaults to "heatmaps_combined.png".
        show_points (bool, optional): Whether to plot maxima and ground truth keypoints. Defaults to True.
        threshold (float, optional): Minimum heatmap max value to include in overlay. Defaults to 0.05.
        keypoints (list of np.ndarray, optional): Ground truth keypoints [num_points, 2]. Defaults to None.
        output_shape (tuple, optional): Model output shape (H,W). If None, inferred from heatmaps.
    """
    # Convert torch tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()
    if torch.is_tensor(keypoints):
        keypoints = keypoints.cpu().numpy()

    # Convert image format to HWC if needed
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    H, W = image.shape[:2]

    # Get model output dimensions for coordinate scaling
    if output_shape is None:
        H_out, W_out = heatmaps.shape[-2:]
    else:
        H_out, W_out = output_shape

    # Combine all keypoint heatmaps for a single overlay
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
        # Mark maxima of each individual heatmap
        for i, hm in enumerate(heatmaps):
            hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
            y, x = np.unravel_index(np.argmax(hm_resized), hm_resized.shape)
            ax.scatter(x, y, c="red", s=6, label="Prediction" if i == 0 else None)

        # Overlay ground truth keypoints
        if keypoints is not None:
            keypoints = np.atleast_2d(keypoints).copy()
            keypoints[:, 0] = keypoints[:, 0] * W / W_out
            keypoints[:, 1] = keypoints[:, 1] * H / H_out
            ax.scatter(keypoints[:, 0], keypoints[:, 1], c="green", s=6, marker="x", label="Annotation")

    # Filter out duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.axis("off")
    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(os.path.join(path_to_save, fname), bbox_inches="tight")
    plt.close(fig)


def visualize_keypoints_combined(image, keypoints_pred, path_to_save, fname="keypoints_combined.png",
                                 show_points=True, keypoints=None, output_shape=None):
    """Visualize predicted keypoints directly overlaid on an image.

    Args:
        image (np.ndarray or torch.Tensor): Input image [C,H,W] or [H,W,3].
        keypoints_pred (np.ndarray or torch.Tensor): Predicted keypoints [N,2] in model output coordinates.
        path_to_save (str): Directory to save the resulting figure.
        fname (str, optional): Filename for the saved figure. Defaults to "keypoints_combined.png".
        show_points (bool, optional): Whether to plot predicted and ground truth keypoints. Defaults to True.
        keypoints (list of np.ndarray, optional): Ground truth keypoints [num_points, 2]. Defaults to None.
        output_shape (tuple, optional): Model output shape (H,W). Defaults to None.
    """
    # Convert torch tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(keypoints_pred):
        keypoints_pred = keypoints_pred.cpu().numpy()
    if torch.is_tensor(keypoints):
        keypoints = keypoints.cpu().numpy()

    # Convert image format to HWC if needed
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    H, W = image.shape[:2]

    keypoints_pred = np.atleast_2d(np.asarray(keypoints_pred, dtype=float))

    # Scale model output coordinates to image coordinates
    if output_shape is not None:
        H_out, W_out = output_shape
        keypoints_pred = keypoints_pred.copy()
        keypoints_pred[:, 0] *= W / W_out
        keypoints_pred[:, 1] *= H / H_out

        if keypoints is not None:
            keypoints = np.atleast_2d(
                np.asarray(keypoints, dtype=float)
            ).copy()
            keypoints[:, 0] *= W / W_out
            keypoints[:, 1] *= H / H_out

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    if show_points:
        # Mark predicted keypoints
        for i, (x, y) in enumerate(keypoints_pred):
            ax.scatter(x, y, c="red", s=6, label="Prediction" if i == 0 else None)

        # Overlay ground truth keypoints
        if keypoints is not None:
            # keypoints = np.atleast_2d(np.asarray(keypoints, dtype=float)).copy()
            # if output_shape is not None:
            #     keypoints[:, 0] = keypoints[:, 0] * W / W_out
            #     keypoints[:, 1] = keypoints[:, 1] * H / H_out
            ax.scatter(keypoints[:, 0], keypoints[:, 1], c="green", s=6, marker="x", label="Annotation")

    # Filter out duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.axis("off")
    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(os.path.join(path_to_save, fname), bbox_inches="tight")
    plt.close(fig)


def output_to_original_coordinates(coordinates, heatmap_shape, original_shape):
    """Converts keypoint coordinates from model output space to original image space.

    Args:
        coordinates (np.ndarray): Keypoint coordinates in model output space [N, 2].
        output_shape (tuple): Model output dimensions (height, width).
        original_shape (tuple): Original image dimensions (height, width).

    Returns:
        np.ndarray: Keypoint coordinates in original image space [N, 2].
    """

    heatmap_height, heatmap_width = heatmap_shape
    original_height, original_width = original_shape

    result = coordinates.copy().astype(float)
    result[:, 0] *= original_width / heatmap_width
    result[:, 1] *= original_height / heatmap_height

    return result


def predict_images(cfg, trainer, loader, metrics, folder_path, annotations_label):
    """Runs the prediction loop over the dataset, visualizes results, computes metrics,
    and creates a DataFrame from the predictions.

    Args:
        cfg (dict): Configuration dictionary containing model, predict, and dataset settings.
        trainer (BaseTrainer): Initialized PyTorch trainer object for running inference.
        loader (generator): Data loader yielding batches of images and annotations.
        metrics (list): List of metric objects to track and compute.
        folder_path (str): Base output directory to save results and visualizations.
        annotations_label (list of str): List of string labels for each keypoint.

    Returns:
        pd.DataFrame: DataFrame containing detailed coordinates for predictions and ground truth.
    """
    path_to_save = os.path.join(folder_path, "prediction")
    predict_images_enabled = cfg["predict"].get("predict_images", True)

    if predict_images_enabled:
        os.makedirs(path_to_save, exist_ok=True)

    results = []
    records = []

    metric_names = [type(m).__name__ for m in metrics]

    # === Prediction loop ===
    predictions_output = trainer.predict(
        loader,
        method=cfg["model"]["predict_type"],
        checkpoint=cfg["predict"]["model"],
        **cfg["dataset"]["params"]
    )

    for batch, preds in predictions_output:
        for img, pred, fname, keypoints, norm_coefficient, orig_height, orig_width in zip(
            batch["image"],
            preds,
            batch["filename"],
            batch["keypoints"],
            batch["norm_coefficient"],
            batch["orig_height"],
            batch["orig_width"]
        ):
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()

            if torch.is_tensor(keypoints):
                keypoints = keypoints.cpu().numpy()

            keypoints = np.asarray(keypoints)

            # Determine the model output resolution used for coordinate conversion.
            # Sliding-window prediction uses window_output_size, while standard prediction uses output_size.
            if cfg["model"]["predict_type"] == "sliding_window":
                heatmap_height, heatmap_width = cfg["dataset"]["params"]["window_output_size"]
            else:
                heatmap_height, heatmap_width = cfg["dataset"]["params"]["output_size"]

            if cfg["keypoint_format"] == "keypoints":
                # DSNT output: [K, 2]
                predictions = np.asarray(pred, dtype=float)

                if cfg["model"]["predict_type"] == "sliding_window":
                    if img.ndim == 3 and img.shape[0] in (1, 3):
                        img_h, img_w = img.shape[1:3]
                    else:
                        img_h, img_w = img.shape[:2]

                    win_size = cfg["dataset"]["params"]["window_size"]
                    out_size = cfg["dataset"]["params"]["window_output_size"]

                    win_h = win_size[0] if isinstance(win_size, (list, tuple)) else win_size
                    win_w = win_size[1] if isinstance(win_size, (list, tuple)) else win_size
                    out_h = out_size[0] if isinstance(out_size, (list, tuple)) else out_size
                    out_w = out_size[1] if isinstance(out_size, (list, tuple)) else out_size

                    heatmap_height = img_h * (out_h / win_h)
                    heatmap_width = img_w * (out_w / win_w)
                else:
                    heatmap_height, heatmap_width = cfg["dataset"]["params"]["output_size"]
                    
            else:
                # Heatmap output: [K, H, W]
                heatmap_height, heatmap_width = pred.shape[-2:]

                predictions = []
                for heatmap in pred:
                    y_pred, x_pred = np.unravel_index(
                        np.argmax(heatmap),
                        heatmap.shape
                    )
                    predictions.append([x_pred, y_pred])

                predictions = np.asarray(predictions, dtype=float)

            # Convert coordinates to original image space
            prediction_original = output_to_original_coordinates(
                predictions,
                (heatmap_height, heatmap_width),
                (orig_height, orig_width)
            )
            annotation_original = output_to_original_coordinates(
                keypoints,
                (heatmap_height, heatmap_width),
                (orig_height, orig_width)
            )

            # Visualize combined heatmap
            if predict_images_enabled:
                if cfg["keypoint_format"] == "keypoints":
                    visualize_keypoints_combined(
                        img,
                        predictions,
                        path_to_save,
                        fname,
                        show_points=True,
                        keypoints=keypoints,
                        output_shape=(heatmap_height, heatmap_width)
                    )
                else:
                    visualize_heatmaps_combined(
                        img,
                        pred,
                        path_to_save,
                        fname,
                        show_points=True,
                        keypoints=keypoints,
                        output_shape=(heatmap_height, heatmap_width)
                    )

            # ============================================================
            # Metrics for current image - HEATMAP SPACE
            # ============================================================

            metrics_heatmap = {}

            for m in metrics:
                m.reset()
                m.update(
                    np.expand_dims(predictions, axis=0),
                    np.expand_dims(keypoints, axis=0),
                    norm_coefficient=norm_coefficient,
                )
                metrics_heatmap[type(m).__name__] = m.compute()

            # ============================================================
            # Metrics for current image - ORIGINAL IMAGE SPACE
            # ============================================================

            metrics_original = {}

            for m in metrics:
                m.reset()
                m.update(
                    np.expand_dims(prediction_original, axis=0),
                    np.expand_dims(annotation_original, axis=0),
                    norm_coefficient=norm_coefficient,
                )
                metrics_original[type(m).__name__] = m.compute()

            # ============================================================
            # Add filename and metrics to results
            # ============================================================

            row = {"filename": fname}

            # Heatmap metrics
            for metric_name in metric_names:
                row[f"{metric_name}"] = metrics_heatmap[metric_name]

            # Original image metrics
            for metric_name in metric_names:
                row[f"{metric_name}_original"] = metrics_original[metric_name]

            results.append(row)

            # ============================================================
            # Add prediction and annotation coordinates to DataFrame
            # ============================================================

            record = {
                "filename": fname,
                "orig_height": int(orig_height),
                "orig_width": int(orig_width),
                "heatmap_height": int(heatmap_height),
                "heatmap_width": int(heatmap_width),
            }

            for keypoint_idx in range(pred.shape[0]):
                label = annotations_label[keypoint_idx]

                record.update({
                    # Heatmap space
                    f"{label}_ann_x": float(keypoints[keypoint_idx, 0]),
                    f"{label}_ann_y": float(keypoints[keypoint_idx, 1]),
                    f"{label}_pred_x": float(predictions[keypoint_idx, 0]),
                    f"{label}_pred_y": float(predictions[keypoint_idx, 1]),

                    # Original image space
                    f"{label}_ann_orig_x": float(annotation_original[keypoint_idx, 0]),
                    f"{label}_ann_orig_y": float(annotation_original[keypoint_idx, 1]),
                    f"{label}_pred_orig_x": float(prediction_original[keypoint_idx, 0]),
                    f"{label}_pred_orig_y": float(prediction_original[keypoint_idx, 1]),
                })

            records.append(record)

    # ============================================================
    # Sort results
    # ============================================================

    # Best/worst selection is based on HEATMAP metrics,
    # same behavior as before.
    for metric, direction in reversed(
        list(zip(metric_names, cfg["predict"]["metrics_directions"]))
    ):
        is_reverse = (direction == '+')
        results.sort(key=lambda x: x[f"{metric}"], reverse=is_reverse)

    # ============================================================
    # Save sorted results to CSV
    # ============================================================

    csv_path = os.path.join(
        folder_path,
        f"{cfg['predict']['prefix']}_metrics_sorted.csv"
    )

    os.makedirs(folder_path, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["filename"]

        # Heatmap metrics
        header.extend([f"{metric}" for metric in metric_names])

        # Original image metrics
        header.extend([f"{metric}_original" for metric in metric_names])

        writer.writerow(header)

        for row in results:
            writer.writerow([row[h] for h in header])

    print(f"Results saved to {csv_path}")

    return pd.DataFrame(records)


def copy_files(rows, base_dir, folder_path, image_column):
    """Copies selected image files from the prediction folder to a new destination.

    Args:
        rows (list of dict): Data rows containing filenames to copy.
        base_dir (str): Target directory to copy the files into.
        folder_path (str): Base output directory where the 'prediction' subfolder is located.
        image_column (str): The dictionary key corresponding to the filename in 'rows'.
    """
    os.makedirs(base_dir, exist_ok=True)

    for row in rows:
        filename = row[image_column]
        src = os.path.join(folder_path, "prediction", filename)

        if not os.path.exists(src):
            print(f"File does not exist: {src}")
            continue

        filename = os.path.basename(src)
        dst = os.path.join(base_dir, filename)

        # Copy original file with metadata
        shutil.copy2(src, dst)


def select_images(cfg, folder_path):
    """Selects and copies the best and worst performing images based on sorted metrics.

    Args:
        cfg (dict): Configuration dictionary containing selection parameters.
        folder_path (str): Base output directory containing the CSV results.
    """
    csv_path = os.path.join(folder_path, f"{cfg['predict']['prefix']}_metrics_sorted.csv")

    image_column = "filename"

    # Output directories for best and worst predictions
    best_base = os.path.join(folder_path, "prediction_selection/best", cfg["predict"]["prefix"])

    worst_base = os.path.join(folder_path, "prediction_selection/worst", cfg["predict"]["prefix"])

    # ==== LOAD CSV ====
    rows = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    # Get top and bottom N rows based on the sorted CSV
    best_rows = rows[:cfg["predict"]["num_select_images"]]
    worst_rows = rows[-cfg["predict"]["num_select_images"]:]

    copy_files(best_rows, best_base, folder_path, image_column)
    copy_files(worst_rows, worst_base, folder_path, image_column)


def save_dataframe_csv(cfg, folder_path, df):
    """Saves the predictions DataFrame to a CSV file.

    Args:
        cfg (dict): Configuration dictionary.
        folder_path (str): Output directory path.
        df (pd.DataFrame): DataFrame containing prediction data.
    """
    output_csv = os.path.join(folder_path, f"{cfg['predict']['prefix']}_predictions_vs_annotations.csv")

    df.to_csv(output_csv, index=False)
    print(f"CSV saved to: {output_csv}")


def save_dataframe_json(cfg, folder_path, df):
    """Saves the predictions DataFrame to a JSON file in a keypoints-per-file format.

    Args:
        cfg (dict): Configuration dictionary.
        folder_path (str): Output directory path.
        df (pd.DataFrame): DataFrame containing prediction data.
    """
    output_json = os.path.join(folder_path, f"{cfg['predict']['prefix']}_keypoints.json")

    results = []

    for _, row in df.iterrows():
        keypoints = {}

        for col in df.columns:
            if "_pred_x" in col and "_pred_orig_x" not in col:
                label = col.replace("_pred_x", "")

                keypoint = {}

                keypoint["heatmap"] = {
                    "x": float(row[f"{label}_pred_x"]),
                    "y": float(row[f"{label}_pred_y"])
                }

                keypoint["original"] = {
                    "x": float(row[f"{label}_pred_orig_x"]),
                    "y": float(row[f"{label}_pred_orig_y"])
                }

                keypoints[label] = keypoint

        results.append({
            "filename": row["filename"],
            "keypoints": keypoints
        })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"JSON saved to: {output_json}")


def compute_keypoint_errors(df, coordinate_space="original"):
    """Computes Euclidean pixel errors between predicted and ground truth keypoints.

    Args:
        df (pd.DataFrame): DataFrame containing prediction and annotation coordinates.
        coordinate_space (str): Coordinate space, either "heatmap" or "original".

    Returns:
        dict: Dictionary mapping keypoint labels to numpy arrays of their errors.
    """
    errors = {}

    suffix = "" if coordinate_space == "heatmap" else "_orig"

    for col in df.columns:
        if col.endswith(f"_ann{suffix}_x"):
            label = col.replace(f"_ann{suffix}_x", "")

            x_ann = df[f"{label}_ann{suffix}_x"]
            y_ann = df[f"{label}_ann{suffix}_y"]
            x_pred = df[f"{label}_pred{suffix}_x"]
            y_pred = df[f"{label}_pred{suffix}_y"]

            err = np.sqrt((x_pred - x_ann) ** 2 + (y_pred - y_ann) ** 2)

            errors[label] = err

    return errors


def analysis_per_keypoint(cfg, errors, output_dir, suffix=""):
    """Calculates basic statistics (mean, median, std, max) for each keypoint.

    Args:
        cfg (dict): Configuration dictionary.
        errors (dict): Dictionary of pixel errors per keypoint.
        output_dir (str): Directory to save the statistics CSV.
        suffix (str): Suffix used for the output filename.

    Returns:
        dict: Nested dictionary containing statistics for each keypoint label.
    """
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

    keypoint_df.to_csv(
        os.path.join(output_dir, f"{cfg['predict']['prefix']}_per_keypoint_stats{suffix}.csv"),
        index=False
    )

    return stats


def analysis_per_image(cfg, df, errors, output_dir, suffix=""):
    """Calculates and saves error statistics aggregated per image.

    Args:
        cfg (dict): Configuration dictionary.
        df (pd.DataFrame): DataFrame containing image filenames.
        errors (dict): Dictionary of keypoint errors.
        output_dir (str): Directory to save the per-image statistics CSV.
        suffix (str): Suffix used for the output filename.

    Returns:
        pd.DataFrame: DataFrame containing mean and max errors per image.
    """
    print("\n=== Per-image analysis ===")

    err_df = pd.DataFrame(errors)

    df_results = pd.DataFrame({
        "filename": df["filename"],
        "mean_error": err_df.mean(axis=1),
        "max_error": err_df.max(axis=1)
    })

    print("\nTop 5 worst images:")
    print(df_results.sort_values("mean_error", ascending=False)[
        ["filename", "mean_error", "max_error"]
    ].head())

    df_results.to_csv(
        os.path.join(output_dir, f"{cfg['predict']['prefix']}_per_image_stats{suffix}.csv"),
        index=False
    )

    return df_results


def analysis_of_mean_error(df, stats):
    """Calculates the overall mean and standard deviation of all keypoint mean errors.

    Args:
        df (pd.DataFrame): Predictions DataFrame.
        stats (dict): Dictionary containing statistics per keypoint.

    Returns:
        tuple: (mean_of_means, std_of_means) representing the global error stats.
    """
    # Extract mean values across all keypoints
    means = [v['mean'] for v in stats.values()]

    # Calculations
    mean_of_means = np.mean(means)
    std_of_means = np.std(means, ddof=1)

    print("mean of means:", mean_of_means)
    print("std of means:", std_of_means)

    return mean_of_means, std_of_means


def analysis_metrics(cfg, df, coordinate_space="heatmap"):
    """Computes advanced metrics (e.g., PCK, AED, Cobb) from keypoint coordinates.

    Args:
        cfg (dict): Configuration dictionary.
        df (pd.DataFrame): DataFrame with keypoint coordinates.
        coordinate_space (str): Coordinate space, either "heatmap" or "original".

    Returns:
        dict: Dictionary containing the computed metric values.
    """
    # --- 1. Find all keypoint labels ---
    keypoints = sorted({
        c.replace("_ann_x", "")
        for c in df.columns if c.endswith("_ann_x")
    })

    B = len(df)
    K = len(keypoints)

    # --- 2. Create target and prediction arrays ---
    preds = np.zeros((B, K, 2))
    targets = np.zeros((B, K, 2))

    if coordinate_space == "heatmap":
        ann_x_suffix = "_ann_x"
        ann_y_suffix = "_ann_y"
        pred_x_suffix = "_pred_x"
        pred_y_suffix = "_pred_y"
    elif coordinate_space == "original":
        ann_x_suffix = "_ann_orig_x"
        ann_y_suffix = "_ann_orig_y"
        pred_x_suffix = "_pred_orig_x"
        pred_y_suffix = "_pred_orig_y"
    else:
        raise ValueError(
            f"Unknown coordinate space: {coordinate_space}"
        )

    for i, kp in enumerate(keypoints):
        targets[:, i, 0] = df[f"{kp}{ann_x_suffix}"]
        targets[:, i, 1] = df[f"{kp}{ann_y_suffix}"]

        preds[:, i, 0] = df[f"{kp}{pred_x_suffix}"]
        preds[:, i, 1] = df[f"{kp}{pred_y_suffix}"]

    # --- 3. Get normalization coefficient from config ---
    norm_coefficient = cfg["dataset"]["params"]["norm_coefficient"]

    # --- 4. Compute metrics ---
    metrics = get_metrics(cfg["predict"]["kp_metrics"])
    metrics_values = {}

    for m in metrics:
        m.reset()
        m.update(
            preds,
            targets,
            norm_coefficient=norm_coefficient
        )

        metrics_values[type(m).__name__] = m.compute()

    print(f"{coordinate_space} metrics:", metrics_values)

    return metrics_values


def plot_distributions(errors, output_dir):
    """Plots the overall error distribution histogram and a boxplot of all keypoints.

    Args:
        errors (dict): Dictionary mapping keypoint labels to error arrays.
        output_dir (str): Directory to save the generated plots.
    """
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
    """Plots individual error distribution histograms for each keypoint.

    Args:
        errors (dict): Dictionary mapping keypoint labels to error arrays.
        output_dir (str): Directory to save the generated histograms.
    """
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
    """Plots a combined boxplot showing the distribution of summary statistics across keypoints.

    Args:
        stats (dict): Dictionary containing summary statistics per keypoint.
        output_dir (str): Directory to save the generated plot.
    """
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
    """Plots individual boxplots for each summary metric (mean, median, std, max) across keypoints.

    Args:
        stats (dict): Dictionary containing summary statistics per keypoint.
        output_dir (str): Directory to save the generated plots.
    """
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
    """Plots bar charts for each summary metric broken down by individual keypoints.

    Args:
        stats (dict): Dictionary containing summary statistics per keypoint.
        output_dir (str): Directory to save the generated bar charts.
    """
    print("\n=== Plotting metrics per keypoint ===")
    os.makedirs(output_dir, exist_ok=True)

    # Seaborn theme + larger fonts
    sns.set_theme(style="whitegrid", context="talk")
    # context="talk" enlarges text (can use "poster" for even larger)

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

        # Enlarge tick labels
        ax.tick_params(axis='both', labelsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_per_keypoint.png"), dpi=300)
        plt.close()


def save_results(cfg, mean, std, metrics_dict, folder_path):
    """Saves final computed metrics and global statistics into a JSON file.

    Args:
        cfg (dict): Configuration dictionary.
        mean (dict): Overall mean of means error for both coordinate spaces.
        std (dict): Overall standard deviation of means error for both coordinate spaces.
        metrics_dict (dict): Dictionary containing advanced metrics results.
        folder_path (str): Output directory path.
    """
    # Convert numpy types to native Python floats for JSON serialization
    metrics_clean = {}

    for coordinate_space, values in metrics_dict.items():
        metrics_clean[coordinate_space] = {
            k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
            for k, v in values.items()
        }

    data = {
        "mean_of_means": {
            "heatmap": float(mean["heatmap"]),
            "original": float(mean["original"])
        },
        "std_of_means": {
            "heatmap": float(std["heatmap"]),
            "original": float(std["original"])
        },
        "metrics": metrics_clean
    }

    path = os.path.join(
        folder_path,
        f"{cfg['predict']['prefix']}_results.json"
    )

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved to {path}")


def analyze_prediction(cfg, folder_path, df):
    """Executes the full analysis pipeline on predictions and generates reports/plots.

    Args:
        cfg (dict): Configuration dictionary.
        folder_path (str): Base output directory.
        df (pd.DataFrame): DataFrame containing prediction coordinates and annotations.
    """
    output_dir = os.path.join(
        folder_path,
        "analysis",
        cfg["predict"]["prefix"]
    )

	# Remove previous analysis results
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # HEATMAP SPACE
    # ============================================================

    print("\n========================================")
    print("ANALYSIS - HEATMAP SPACE")
    print("========================================")

    errors_heatmap = compute_keypoint_errors(df, coordinate_space="heatmap")

    stats_heatmap = analysis_per_keypoint(cfg, errors_heatmap, folder_path, suffix="_heatmap")

    analysis_per_image(cfg, df, errors_heatmap, folder_path, suffix="_heatmap")

    mean_heatmap, std_heatmap = analysis_of_mean_error(df, stats_heatmap)

    metrics_heatmap = analysis_metrics(cfg, df, coordinate_space="heatmap")

    heatmap_output_dir = os.path.join( output_dir, "heatmap")

    os.makedirs(heatmap_output_dir, exist_ok=True)

    plot_distributions(errors_heatmap, heatmap_output_dir)

    plot_keypoint_distributions(errors_heatmap, heatmap_output_dir)

    plot_keypoint_stats_boxplot( stats_heatmap, heatmap_output_dir)

    plot_metrics_per_keypoint( stats_heatmap, heatmap_output_dir)

    # ============================================================
    # ORIGINAL IMAGE SPACE
    # ============================================================

    print("\n========================================")
    print("ANALYSIS - ORIGINAL IMAGE SPACE")
    print("========================================")

    errors_original = compute_keypoint_errors(df, coordinate_space="original")

    stats_original = analysis_per_keypoint(cfg, errors_original, folder_path, suffix="_original")

    analysis_per_image(cfg, df, errors_original, folder_path, suffix="_original")

    mean_original, std_original = analysis_of_mean_error(df, stats_original)

    metrics_original = analysis_metrics(cfg, df, coordinate_space="original")

    original_output_dir = os.path.join( output_dir, "original")

    os.makedirs(original_output_dir, exist_ok=True)

    plot_distributions( errors_original, original_output_dir)

    plot_keypoint_distributions(errors_original, original_output_dir)

    plot_keypoint_stats_boxplot(stats_original, original_output_dir)

    plot_metrics_per_keypoint(stats_original, original_output_dir)

    # ============================================================
    # SAVE FINAL RESULTS
    # ============================================================

    mean = {
        "heatmap": mean_heatmap,
        "original": mean_original
    }

    std = {
        "heatmap": std_heatmap,
        "original": std_original
    }

    metrics = {
        "heatmap": metrics_heatmap,
        "original": metrics_original
    }

    save_results(cfg, mean, std, metrics, folder_path)

    print("\nDone. Outputs are located in:", output_dir)


def main():
    """Main function to load configuration, initialize components, run predictions, and perform analysis."""
    # Load config and initialize necessary components
    cfg = load_config("configs/config_list.yaml")
    print("Configuration loaded.")
    print(cfg)

    # Create model
    model = get_model(
        cfg["model"]["name"],
        use_dsnt=cfg["model"]["use_dsnt"], 
        cfg=cfg,
        **cfg["model"]["params"]
    )

    print(f"Model '{cfg['model']['name']}' created.")

    # Load dataset
    transform = get_transform(
        transform=None,
        format=cfg["keypoint_format"],
        predict_type=f"{cfg['model']['predict_type']}_val",
        **cfg["dataset"]["params"]
    )

    dataset = get_dataset(
        name=cfg["predict"]["name"],
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        keypoint_format=cfg["keypoint_format"],
        transform=transform,
        **cfg["dataset"]["params"]
    )

    print(f"Dataset loaded: {len(dataset)} images.")

    # Load annotation labels from text file
    txt_path = os.path.join(
        cfg["dataset"]["params"]["root_dir"],
        cfg["predict"]["annotation_label"]
    )

    annotations_label = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            if line:
                annotations_label.append(line)

    print("Annotation labels:", annotations_label)

    # Wrap dataset in batch loader
    loader = batch_loader(
        dataset,
        batch_size=cfg["predict"]["batch_size"]
    )

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
        keypoint_format=cfg["keypoint_format"],
        special_mode=cfg["model"].get("special_mode"),
        **cfg["train"]
    )

    print("Trainer initialized.")

    folder_path = os.path.join(
        "results",
        cfg["experiment"]["name"]
    )

    predict_images_enabled = cfg["predict"]["predict_images"]

    select_images_enabled = cfg["predict"]["select_images"]

    save_predictions_enabled = cfg["predict"]["save_predictions"]

    # Predict on images
    df = predict_images(
        cfg,
        trainer,
        loader,
        metrics,
        folder_path,
        annotations_label
    )

    # Save best/worst images
    if predict_images_enabled and select_images_enabled:
        select_images(cfg, folder_path)

    # Save prediction coordinates and analyze predictions
    if save_predictions_enabled:
        # Save to CSV
        save_dataframe_csv(cfg, folder_path, df)

        # Save to JSON
        save_dataframe_json(cfg, folder_path, df)

        # Analyze predictions
        analyze_prediction(cfg, folder_path, df)


if __name__ == "__main__":
    main()