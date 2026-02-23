import torch
from factory import ModelFactory, ModelType
from data.util import *
from training.util import set_random_seed
from evaluation.util import visualize_predictions
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = False
MODEL_TYPE = ModelType.RETINANET
EXT = False
BATCH_SIZE = 1
NUM_WORKERS = 0
IMG_SIZE = 640
NUM_QUERIES = 100
CONF_THRESH = 0.5
IOU_THRESH = 0.45
NUM_CLASSES = 7


def main():
    set_random_seed(42)

    # 1. Load Model
    factory = ModelFactory(device=DEVICE)
    model = factory.load(model_type=MODEL_TYPE, num_queries=NUM_QUERIES)

    # 2. Load weights
    pretrained_str = "pretrained" if PRETRAINED else "from_scratch"
    ckpt = torch.load(
        f"checkpoints/{pretrained_str}/{MODEL_TYPE}_model.pt",
        map_location=DEVICE,
    )
    model.load_state_dict(ckpt)
    model.eval()

    print(f"✅ Loaded {MODEL_TYPE} model ({pretrained_str}) for inference.")

    # 3. Load sample data
    dataset = "avian_malaria" if EXT else "vogelbacher23"
    data_path = "data/preprocessed/" + dataset
    test_loader = load_dataloader(
        dataset_type=DatasetType.EXT if EXT else DatasetType.TEST,
        data_path=data_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )

    loader_iter = iter(test_loader)
    images, targets = next(loader_iter)

    images = images.to(DEVICE)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    print("✅ Loaded test data for inference.")

    # 4. Run inference
    with torch.no_grad():
        out = model.predict(images, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH)
        print(out)
    print("✅ Inference completed.")

    # 5. Visualize predictions and targets
    for i, (image, prediction, target) in enumerate(
        zip(images, out["predictions"], targets)
    ):
        fig = visualize_predictions(
            image,
            prediction,
            target=target,
            class_visualization=False,
            show_targets=False,
        )
        output_dir = os.path.join("results", "predictions", pretrained_str)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{MODEL_TYPE}_prediction.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

        print(f"✅ Saved prediction visualization for image {i}.")


if __name__ == "__main__":
    main()
