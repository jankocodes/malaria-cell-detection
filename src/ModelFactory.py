import torch

# Import all necessary libraries for the models
from ultralytics import YOLO  # For YOLOv8
from torchvision.models.detection import retinanet_resnet50_fpn

# import yolov5 # For YOLOv5 (if using PyTorch Hub or custom repo)
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection import RetinaNet
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from yolov5.models.common import DetectMultiBackend


class ModelLoader:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Model factory initialized. Using device: {self.device}")

    def load_yolo_v8(
        self, pretrained=False, model_name="yolov8s", model_dir="models/yolov8"
    ) -> YOLO:

        model_path = Path(model_dir) / f"{model_name}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if pretrained:
            model = YOLO(str(model_path))

        else:
            yaml_path = Path(model_dir) / f"{model_name}.yaml"
            if not yaml_path.exists():

                model = YOLO(f"{model_name}.yaml")
            else:
                model = YOLO(str(yaml_path))

        return model

    def load_yolo_v5(
        self,
        repo_path: str = "ultralytics/yolov5",
        pretrained: bool = True,
        model_name: str = "yolov5s",
        weight_path: str = "models/yolov5",
    ) -> DetectMultiBackend:

        if pretrained:
            model = torch.hub.load(
                repo_path,
                "custom",
                autoshape=False,
                path=f"{weight_path}/{model_name}.pt",
            )

        else:
            model = torch.hub.load(
                repo_path, model_name, pretrained=False, autoshape=False
            )

        return model

    def load_retinanet_pipeline(
        self, pretrained=True, weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
    ) -> RetinaNet:

        preprocess = weights.transforms()

        model = retinanet_resnet50_fpn(
            weights=weights if pretrained else None, score_thresh=0.7
        )

        return (preprocess, model)

    def load_detr_pipeline(
        self, pretrained=True, weights="facebook/detr-resnet-50"
    ) -> tuple[DetrImageProcessor, DetrForObjectDetection]:

        processor = (
            DetrImageProcessor.from_pretrained(weights)
            if pretrained
            else DetrImageProcessor()
        )

        if pretrained:
            model = DetrForObjectDetection.from_pretrained(weights)
        else:
            model = DetrForObjectDetection(DetrForObjectDetection.config_class())

        return (processor, model)
