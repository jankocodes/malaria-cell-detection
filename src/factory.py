import sys
from pathlib import Path

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


class ModelFactory:
    """
    Lightweight factory for loading object-detection models.

    Supports YOLOv8, YOLOv5, RetinaNet, and DETR architectures.
    Methods return either model instances or (preprocessor, model) tuples.
    Models are not automatically moved to the specified device.

    Attributes:
        device (str): The device to use for model inference ('cuda' or 'cpu').
                     Defaults to 'cuda' if available, otherwise 'cpu'.

    Methods:
        load_yolo_v8(pretrained, model_name, model_dir) -> YOLO:
            Load a YOLOv8 model. If pretrained=True, loads weights from model_dir.
            Otherwise, loads from YAML configuration file.

        load_yolo_v5(repo_path, pretrained, model_name, weight_path) -> DetectMultiBackend:
            Load a YOLOv5 model from torch.hub. If pretrained=True, loads custom weights
            from weight_path. Otherwise, loads the base model without pretrained weights.

        load_retinanet_pipeline(pretrained, weights) -> tuple[Callable, RetinaNet]:
            Load a RetinaNet model with its preprocessing pipeline.
            Returns a tuple of (preprocessor, model).

        load_detr_pipeline(pretrained, weights) -> tuple[DetrImageProcessor, DetrForObjectDetection]:
            Load a DETR model with its image processor.
            Returns a tuple of (processor, model).
    """

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
        num_classes: int = 7,
    ) -> DetectMultiBackend:

        model = torch.hub.load(
            repo_path,
            "custom",
            autoshape=False,
            path=f"{weight_path}/{model_name}.pt",
        )

        if not pretrained:
            print("⚠ Resetting all YOLOv5 model weights to random initialization...")
            model.model.apply(self._reset_weights)

        # -----------------------------
        # 🔥 MODIFY NUMBER OF CLASSES
        # -----------------------------
        model.nc = num_classes
        model.names = [f"class_{i}" for i in range(num_classes)]

        # Update Detect() module inside model.model
        detect_layer = model.model.model[-1]  # YOLO head is always the last layer

        # Set new class count
        detect_layer.nc = num_classes
        detect_layer.no = num_classes + 5  # x,y,w,h,obj + classes

        # Re-create conv layers for each detection head
        detect_layer.m = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    x.in_channels, detect_layer.no * len(detect_layer.anchors[i]), 1
                )
                for i, x in enumerate(detect_layer.m)
            ]
        )

        # Reset bias/weights
        for m in detect_layer.m:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(m.bias)

        return model

    def _reset_weights(self, m):
        """Reset weights of Conv, BN, Linear, etc."""
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

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
