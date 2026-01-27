import os
import json
import numpy as np


def save_ap_results(
    metrics, model_type, pretrained_str, data_path, num_img, num_classes, iou_theshold
):
    # Output path with pretrained subfolder
    output_dir = os.path.join("results", "eval", pretrained_str)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{model_type}_ap{iou_theshold}.json")

    # Build results dictionary
    results = {
        "model_type": model_type,
        "pretrained": pretrained_str,
        "dataset": data_path,
        "num_images": num_img,
        "iou_threshold": iou_theshold,
        "mAP@0.5": float(metrics["mAP"]),
        "per_class_ap": {},
    }

    # Per-class AP
    for i in range(num_classes):
        ap_value = metrics[f"AP_class_{i}"]
        results["per_class_ap"][f"class_{i+1}"] = (
            None if np.isnan(ap_value) else float(ap_value)
        )

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to: {output_path}")
