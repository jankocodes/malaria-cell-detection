import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from factory import ModelFactory, ModelType
from data.util import *
from training.util import set_random_seed
from evaluation.metrics import calculate_ap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = True
MODEL_TYPE = ModelType.YOLOV5
DATASET_TYPE = DatasetType.TEST
PRETRAINED_STR = "pretrained" if PRETRAINED else "from_scratch"
DATA_PATH = "data/preprocessed/vogelbacher23"
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_CLASSES = 7

if __name__ == "__main__":
    set_random_seed(42)

    # 1. Load Model
    factory = ModelFactory(device=DEVICE)

    model = factory.load(
        model_type=MODEL_TYPE,
    )

    # 2. Load weights
    ckpt = torch.load(
        f"checkpoints/{PRETRAINED_STR}/{MODEL_TYPE}_model.pt", map_location=DEVICE
    )
    model.load_state_dict(ckpt)
    model.eval()

    print(f"✅ Loaded {MODEL_TYPE} model ({PRETRAINED_STR}) for evaluation.")

    # 3. Load test data
    test_loader = load_dataloader(
        dataset_type=DATASET_TYPE,
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=640,
    )

    print(f"✅ Loaded test data: {len(test_loader)} batches")

    # 4. Run evaluation on entire test set
    all_predictions = []
    all_targets = []

    print("\nRunning inference on test set...")
    with torch.no_grad():
        # for batch_idx, (images, targets) in enumerate(test_loader):
        #     images = images.to(DEVICE)
        #     targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        #     # Get predictions
        #     out = model(images)
        #     predictions = out["predictions"]

        #     # Store predictions and targets
        #     all_predictions.extend(predictions)
        #     all_targets.extend(targets)

        #     if (batch_idx + 1) % 10 == 0:
        #         print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
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
        iou_threshold=0.5,
    )

    # 6. Display results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {MODEL_TYPE} ({PRETRAINED_STR})")
    print(f"Dataset: {DATA_PATH}")
    print(f"Num Images: {len(all_predictions)}")
    print(f"IoU Threshold: 0.5")
    print("-" * 50)
    print(f"mAP@0.5: {metrics['mAP']:.4f}")
    print("-" * 50)
    print("Per-class AP:")
    for i in range(NUM_CLASSES):
        ap_value = metrics[f"AP_class_{i}"]
        if not np.isnan(ap_value):
            print(f"  Class {i}: {ap_value:.4f}")
        else:
            print(f"  Class {i}: N/A (no ground truth)")
    print("=" * 50)
