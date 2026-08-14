import json
import os

import pandas as pd

from experiment_config import DATA_PATHS, EXPERIMENTS_BY_MODEL, RESULTS_DIR
from methods.attention_unet import AttentionUNetMethod
from methods.dlv3plus import DLv3Plus
from methods.unet import UNetMethod
from utils.visualization import (
    save_grouped_ablation_chart,
    save_history_plot,
    save_metrics_bar_chart,
    save_prediction_samples,
)


METHOD_REGISTRY = {
    "UNetMethod": UNetMethod,
    "AttentionUNetMethod": AttentionUNetMethod,
    "DLv3Plus": DLv3Plus,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_method(method_class_name, kwargs):
    method_class = METHOD_REGISTRY[method_class_name]
    return method_class(**kwargs)


def save_metrics_json(path, metrics):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def build_comparison_table(results):
    # Report the test-set metrics that are compared in the final report.
    rows = []
    for result in results:
        test_metrics = result["test_metrics"]
        rows.append(
            {
                "model": result["model"],
                "f1_plants": test_metrics.get("f1 - class: plants"),
                "iou_plants": test_metrics.get("IoU - class: plants"),
                "precision_plants": test_metrics.get("precision - class: plants"),
                "recall_plants": test_metrics.get("recall - class: plants"),
                "accuracy": test_metrics.get("accuracy: "),
            }
        )
    return pd.DataFrame(rows)


def run_single_run(model_title, method_class_name, run_config, output_dir):
    ensure_dir(output_dir)
    resolved_method_class_name = run_config.get("method_class", method_class_name)
    method = build_method(resolved_method_class_name, run_config["kwargs"])

    print("-" * 60)
    print(model_title)
    print(run_config["display_name"])

    # Every run uses the same train / validation / test protocol for a fair comparison.
    train_metrics, val_metrics = method.train(
        train_path=DATA_PATHS["train"],
        val_path=DATA_PATHS["validation"],
    )
    test_metrics, _, _ = method.test(test_path=DATA_PATHS["test"])

    method.save_history_csv(os.path.join(output_dir, "history.csv"))
    save_history_plot(
        history=method.history,
        output_path=os.path.join(output_dir, "training_curves.png"),
        title=run_config["display_name"],
    )
    save_prediction_samples(
        method=method,
        test_path=DATA_PATHS["test"],
        output_dir=os.path.join(output_dir, "predictions"),
        max_samples=5,
    )

    save_metrics_json(os.path.join(output_dir, "train_metrics.json"), train_metrics)
    save_metrics_json(os.path.join(output_dir, "val_metrics.json"), val_metrics)
    save_metrics_json(os.path.join(output_dir, "test_metrics.json"), test_metrics)

    return {
        "model": run_config["display_name"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_model_sections(model_config):
    model_root = os.path.join(RESULTS_DIR, model_config["model_key"])
    ensure_dir(model_root)

    model_results = {}

    for section in model_config["sections"]:
        section_root = os.path.join(model_root, section["section_key"])
        ensure_dir(section_root)

        section_results = []
        for run in section["runs"]:
            run_dir = os.path.join(section_root, run["tag"])
            result = run_single_run(
                model_title=model_config["model_title"],
                method_class_name=model_config["method_class"],
                run_config=run,
                output_dir=run_dir,
            )
            section_results.append(result)

        comparison_df = build_comparison_table(section_results)
        comparison_df.to_csv(os.path.join(section_root, "summary.csv"), index=False)

        if len(section_results) == 1:
            save_metrics_bar_chart(
                comparison_df=comparison_df,
                output_path=os.path.join(section_root, "summary.png"),
            )
        else:
            save_grouped_ablation_chart(
                comparison_df=comparison_df,
                output_path=os.path.join(section_root, "summary.png"),
                title=f"{model_config['model_title']} - {section['section_title']}",
            )

        model_results[section["section_title"]] = comparison_df

    return model_results


def run_all_model_experiments():
    ensure_dir(RESULTS_DIR)
    all_results = {}

    for model_config in EXPERIMENTS_BY_MODEL:
        all_results[model_config["model_title"]] = run_model_sections(model_config)

    return all_results
