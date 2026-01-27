import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from factory import ModelFactory, ModelType
from data.util import *
from training.util import set_random_seed

# -------------------------
# Configuration
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = True
MODEL_TYPE = ModelType.DETR
PRETRAINED_STR = "pretrained" if PRETRAINED else "from_scratch"
DATA_PATH = "data/preprocessed/vogelbacher23"
BATCH_SIZE = 16
NUM_WORKERS = 0
IMG_SIZE = 640
NUM_QUERIES = 300

NUM_CLASSES = 7

# -------------------------
# Class color mapping
# -------------------------
CLASS_COLORS = {
    0: (1.0, 0.0, 0.0),  # red
    1: (0.0, 1.0, 0.0),  # green
    2: (0.0, 0.0, 1.0),  # blue
    3: (1.0, 1.0, 0.0),  # yellow
    4: (1.0, 0.0, 1.0),  # magenta
    5: (0.0, 1.0, 1.0),  # cyan
    6: (1.0, 0.5, 0.0),  # orange
}
# Optional: replace with real class names if you have them
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_random_seed(42)

    # 1. Load Model
    factory = ModelFactory(device=DEVICE)
    model = factory.load(model_type=MODEL_TYPE, num_queries=NUM_QUERIES)

    # 2. Load weights
    ckpt = torch.load(
        f"checkpoints/{PRETRAINED_STR}/{MODEL_TYPE}_model.pt",
        map_location=DEVICE,
    )
    model.load_state_dict(ckpt)
    model.eval()

    print(f"✅ Loaded {MODEL_TYPE} model ({PRETRAINED_STR}) for inference.")

    # 3. Load sample data
    # test_loader = load_dataloader(
    #     dataset_type=DatasetType.TEST,
    #     data_path=DATA_PATH,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     img_size=IMG_SIZE,
    # )

    # loader_iter = iter(test_loader)
    # images, targets = next(loader_iter)

    # images = images.to(DEVICE)
    # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    image = load_sample_image(
        DATA_PATH,
        split="test",
        img_id=143,
    )
    print("✅ Loaded test data for inference.")

    # 4. Run inference
    with torch.no_grad():
        out = model(image)

    print("Inference output:", out)
    print(f"Num boxes: {out['predictions'][0]['boxes'].shape[0]}")
    print("✅ Inference completed.")

    # 5. Visualize predictions
    for i, (image, prediction) in enumerate(zip(image, out["predictions"])):
        fig, ax = plt.subplots(1, figsize=(12, 8))

        # Convert image tensor to numpy (assumes already in [0,1])
        img_np = image.cpu().permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            label = int(label)
            color = CLASS_COLORS[label]

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"{CLASS_NAMES[label]} | {score:.2f}",
                color=color,
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.4, pad=1),
            )

        ax.set_title(f"Predictions - Image {i}")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"prediction_visualization_{i}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

        print(f"✅ Saved prediction visualization for image {i}.")
