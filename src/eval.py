import torch
import numpy as np
from factory import ModelFactory, ModelType
from data.util import *
from training.util import set_random_seed
from evaluation.metrics import calculate_ap
from evaluation.util import save_ap_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = True
MODEL_TYPE = ModelType.RETINANET
EXT = False
BATCH_SIZE = 1
NUM_CLASSES = 7
PROCESS_ALL = False  # If False, process only one batch for quick evaluation
AP_IOU_THRESH = 0.34


def main():
    set_random_seed(42)

    # 1. Load Model
    factory = ModelFactory(device=DEVICE)

    model = factory.load(
        model_type=MODEL_TYPE,
    )

    # 2. Load weights
    pretrained_str = "pretrained" if PRETRAINED else "from_scratch"
    ckpt = torch.load(
        f"checkpoints/{pretrained_str}/{MODEL_TYPE}_model.pt", map_location=DEVICE
    )
    model.load_state_dict(ckpt)
    model.eval()

    print(f"✅ Loaded {MODEL_TYPE} model ({pretrained_str}) for evaluation.")

    # 3. Load test data
    dataset = "avian_malaria" if EXT else "vogelbacher23"
    data_path = "data/preprocessed/" + dataset
    test_loader = load_dataloader(
        dataset_type=DatasetType.EXT if EXT else DatasetType.TEST,
        data_path=data_path,
        batch_size=BATCH_SIZE,
        num_workers=0,
        img_size=640,
    )

    # 4. Run evaluation on entire test set
    all_predictions = []
    all_targets = []

    print("\nRunning inference on test set...")
    with torch.no_grad():
        if PROCESS_ALL:
            print(f"✅ Loaded test data: {len(test_loader)} batches")

            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(DEVICE)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                # Get predictions
                out = model(images)
                predictions = out["predictions"]

                # Store predictions and targets
                all_predictions.extend(predictions)
                all_targets.extend(targets)

                print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
        else:
            images, targets = next(iter(test_loader))
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            # Get predictions
            out = model(images)
            predictions = out["predictions"]
            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    print(f"✅ Inference completed on {len(all_predictions)} images.")

    # 5. Calculate AP metrics at IoU=0.5 (AP50)
    print("\nCalculating Average Precision at IoU=0.5...")
    metrics = calculate_ap(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=NUM_CLASSES,
        iou_threshold=AP_IOU_THRESH,
    )

    if PROCESS_ALL:
        save_ap_results(
            metrics=metrics,
            model_type=MODEL_TYPE,
            pretrained_str=pretrained_str,
            data_path=data_path,
            num_img=len(all_predictions),
            num_classes=NUM_CLASSES,
        )

    # 6. Display results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {MODEL_TYPE} ({pretrained_str})")
    print(f"Dataset: {data_path}")
    print(f"Num Images: {len(all_predictions)}")
    print(f"IoU Threshold: {AP_IOU_THRESH}")
    print("-" * 50)
    print(f"mAP@{AP_IOU_THRESH}: {metrics['mAP']:.4f}")
    print("-" * 50)
    print("Per-class AP:")
    for i in range(NUM_CLASSES):
        ap_value = metrics[f"AP_class_{i}"]
        if not np.isnan(ap_value):
            print(f"  Class {i+1}: {ap_value:.4f}")
        else:
            print(f"  Class {i+1}: N/A (no ground truth)")
    print("=" * 50)


if __name__ == "__main__":
    main()
