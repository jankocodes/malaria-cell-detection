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


def visualize_predictions(
    image, prediction, target=None, class_visualization=False, show_targets=True
):
    CLASS_COLORS = {
        0: (1.0, 0.0, 0.0),  # red
        1: (0.0, 1.0, 0.0),  # green
        2: (0.0, 0.0, 1.0),  # blue
        3: (1.0, 1.0, 0.0),  # yellow
        4: (1.0, 0.0, 1.0),  # magenta
        5: (0.0, 1.0, 1.0),  # cyan
        6: (1.0, 0.5, 0.0),  # orange
    }

    _, ax = plt.subplots(1, figsize=(12, 8))

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

        # Determine prediction color
        if class_visualization or target is None:
            prediction_color = CLASS_COLORS[label]
        else:
            # Find matching ground truth box and check if label is correct
            prediction_color = "red"  # Default to red (incorrect)

            target_boxes = target["boxes"].cpu().numpy()
            target_labels = target["class_labels"].cpu().numpy()

            # Find best matching ground truth box using IoU
            best_iou = 0
            best_target_label = None

            for target_box, target_label in zip(target_boxes, target_labels):
                # Calculate IoU
                target_x1, target_y1, target_w, target_h = target_box
                target_x2 = target_x1 + target_w
                target_y2 = target_y1 + target_h

                # Intersection
                inter_x1 = max(x1, target_x1)
                inter_y1 = max(y1, target_y1)
                inter_x2 = min(x2, target_x2)
                inter_y2 = min(y2, target_y2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    pred_area = width * height
                    target_area = target_w * target_h
                    union_area = pred_area + target_area - inter_area
                    iou = inter_area / union_area

                    if iou > best_iou:
                        best_iou = iou
                        best_target_label = int(target_label)

            # If best match has IoU > 0.5 and labels match, color is green
            if best_iou > 0.5 and best_target_label == label:
                prediction_color = "lawngreen"

        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=6,
            edgecolor=prediction_color,
            facecolor="none",
        )

        if target is None:
            ax.text(
                x1,
                y1 - 5,
                f"{score:.2f}",
                color=prediction_color,
                fontsize=10,
                weight="bold",
            )

        ax.add_patch(rect)

    # Plot ground truth targets in green dashed boxes if provided
    if target is not None:
        target_boxes = target["boxes"].cpu().numpy()
        target_labels = target["class_labels"].cpu().numpy()

        for box, label in zip(target_boxes, target_labels):
            if not show_targets:
                continue
            x1, y1, width, height = box

            label = int(label)

            target_color = CLASS_COLORS[label] if class_visualization else "black"
            # Ground truth boxes in white with dashed line
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor=target_color,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

    #        ax.set_title(f"Predictions (red) vs Ground Truth (green dashed) - Image")

    ax.axis("off")

    plt.tight_layout()
    return plt.gcf()  # Return the figure for further processing (e.g., saving)


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
