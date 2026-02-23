import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_ap_results(
    metrics,
    model_type,
    external,
    pretrained_str,
    data_path,
    num_img,
    num_classes,
    iou_theshold,
):
    split = "external" if external else "internal"

    # Output path with pretrained subfolder
    output_dir = os.path.join("results", "eval", split, pretrained_str)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, f"{model_type}_ap{iou_theshold*100:.0f}.json"
    )

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


def visualize_predictions(image, prediction, target=None):
    CLASS_COLORS = {
        0: (1.0, 0.0, 0.0),  # red
        1: (0.0, 1.0, 0.0),  # green
        2: (0.0, 0.0, 1.0),  # blue
        3: (1.0, 1.0, 0.0),  # yellow
        4: (1.0, 0.0, 1.0),  # magenta
        5: (0.0, 1.0, 1.0),  # cyan
        6: (1.0, 0.5, 0.0),  # orange
    }

    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Convert image tensor to numpy (assumes already in [0,1])
    img_np = image.cpu().permute(1, 2, 0).numpy()
    ax.imshow(img_np)

    # Plot predictions in class colors
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        label = int(label)
        color = CLASS_COLORS[label] if target is None else "red"
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )

        if target is None:
            ax.text(
                x1,
                y1 - 5,
                f"{score:.2f}",
                color=color,
                fontsize=10,
                weight="bold",
            )

        ax.add_patch(rect)
    ax.set_title(f"Predictions - Image")

    # Plot ground truth targets in green dashed boxes if provided
    if target is not None:
        target_boxes = target["boxes"].cpu().numpy()
        target_labels = target["class_labels"].cpu().numpy()

        for box, label in zip(target_boxes, target_labels):
            x1, y1, width, height = box

            label = int(label)

            # Ground truth boxes in white with dashed line
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

        ax.set_title(f"Predictions (red) vs Ground Truth (green dashed) - Image")

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"prediction_visualization.png",
        dpi=150,
        bbox_inches="tight",
    )


def enforce_square_box(box, scaling_factor=1.05):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    side_length = max(w, h) * scaling_factor
    x1 -= (side_length - w) / 2
    y1 -= (side_length - h) / 2
    x2 = x1 + side_length
    y2 = y1 + side_length
    return [x1, y1, x2, y2]
