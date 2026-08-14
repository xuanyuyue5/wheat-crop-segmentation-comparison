import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from helpers.helper_fcts import EWS_Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_history_plot(history, output_path, title):
    if not history:
        return

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_loss"], marker="o")
    axes[0].set_title(f"{title} - Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(df["epoch"], df["val_iou"], marker="o", color="tab:green")
    axes[1].set_title(f"{title} - Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar_chart(comparison_df, output_path):
    metrics = ["f1_plants", "iou_plants", "precision_plants", "recall_plants", "accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))

    for axis, metric in zip(axes, metrics):
        axis.bar(comparison_df["model"], comparison_df[metric], color=["#457b9d", "#e76f51", "#2a9d8f"])
        axis.set_title(metric)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_grouped_ablation_chart(comparison_df, output_path, title):
    metrics = ["f1_plants", "iou_plants", "accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))

    for axis, metric in zip(axes, metrics):
        axis.bar(comparison_df["model"], comparison_df[metric], color=["#264653", "#2a9d8f", "#e76f51", "#457b9d"])
        axis.set_title(metric)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def save_prediction_samples(method, test_path, output_dir, max_samples=5):
    ensure_dir(output_dir)

    test_paths = sorted(
        [
            os.path.join(test_path, filename)
            for filename in os.listdir(test_path)
            if filename.endswith("6.png")
        ]
    )
    if not test_paths:
        return

    dataset = EWS_Dataset(test_paths[:max_samples], method.get_transforms(testval=True))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    method.load_model()
    method.model.eval()
    method.model.eval_size()

    with torch.no_grad():
        for index, (X, Y) in enumerate(dataloader):
            X_device = {key: X[key].to(method.device) for key in ["image"]}
            with autocast():
                outputs = method.model(X_device)

            logits = outputs["mask"][:, :, :350, :350]
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            gt_mask = Y["mask"].squeeze(0).squeeze(0).cpu().numpy()
            image = denormalize_image(X["image"].squeeze(0)[:, :350, :350])

            figure, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image)
            axes[0].set_title("Image")
            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Prediction")

            for axis in axes:
                axis.axis("off")

            figure.tight_layout()
            figure.savefig(os.path.join(output_dir, f"sample_{index + 1}.png"), dpi=200, bbox_inches="tight")
            plt.close(figure)
