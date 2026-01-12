import yaml
from pathlib import Path

# Import all necessary libraries for the models
from ultralytics import YOLO  # For YOLOv8
from torchvision.models.detection import retinanet_resnet50_fpn

# import yolov5 # For YOLOv5 (if using PyTorch Hub or custom repo)
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from pathlib import Path
from transformers import DetrConfig, DetrForObjectDetection
import torch
from ultralytics.nn.tasks import DetectionModel
from wrappers.yolov5 import YOLOv5Wrapper
from wrappers.retinanet import RetinaNetWrapper
from wrappers.detr import DetrWrapper
from wrappers.yolov8 import YOLOv8Wrapper
from wrappers.base import BaseDetectionModel
from typing import Any

from enum import Enum


class ModelType(str, Enum):
    YOLOV8 = "yolov8"
    YOLOV5 = "yolov5"
    RETINANET = "retinanet"
    DETR = "detr"


class ModelFactory:
    """
    Factory class for loading and initializing object detection models.

    Supports loading pretrained or custom-trained models from four architectures:
    - YOLOv8
    - YOLOv5
    - RetinaNet
    - DETR

    Each model is wrapped in a corresponding wrapper class that provides a unified interface.

    Attributes:
        device (str): The computing device ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        num_classes (int): Number of object detection classes. Defaults to 7.

    Example:
        >>> factory = ModelFactory(num_classes=7, device='cuda')
        >>> model = factory.load(ModelType.YOLOV8, model_name='yolov8s')
        >>> predictions = model.predict(image)
    """ """
    Factory class for loading and initializing object detection models.
    
    Supports loading pretrained or custom-trained models from four architectures:
    - YOLOv8
    - YOLOv5
    - RetinaNet
    - DETR
    
    Each model is wrapped in a corresponding wrapper class that provides a unified interface.
    
    Attributes:
        device (str): The computing device ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        num_classes (int): Number of object detection classes. Defaults to 7.
    
    Example:
        >>> factory = ModelFactory(num_classes=7, device='cuda')
        >>> model = factory.load(ModelType.YOLOV8, model_name='yolov8s')
        >>> predictions = model.predict(image)
    """

    def __init__(
        self,
        num_classes: int = 7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.num_classes = num_classes

        print(f"Model factory initialized. Using device: {self.device}")

    def load(
        self,
        model_type: ModelType,
        model_dir: str = None,
        **kwargs: Any,
    ) -> BaseDetectionModel:
        """
        Load a model by type using the corresponding factory method.
        """

        if model_type == ModelType.YOLOV8:
            return self.load_yolo_v8(
                model_dir=model_dir if model_dir is not None else "models/yolov8",
                **kwargs,
            )

        elif model_type == ModelType.YOLOV5:
            return self.load_yolo_v5(
                model_dir=model_dir if model_dir is not None else "models/yolov5",
                **kwargs,
            )

        elif model_type == ModelType.RETINANET:
            return self.load_retinanet_pipeline(**kwargs)

        elif model_type == ModelType.DETR:
            return self.load_detr_pipeline(**kwargs)

        else:
            # This should never happen, but is nice for defensive programming
            raise NotImplementedError(f"Model type {model_type} not implemented.")

    def load_yolo_v8(
        self,
        pretrained: bool = True,
        model_name: str = "yolov8s",
        model_dir: str = "models/yolov8",
    ) -> YOLOv8Wrapper:

        weights_path = Path(model_dir) / f"{model_name}.pt"
        yaml_path = Path(model_dir) / f"{model_name}.yml"
        model = DetectionModel(yaml_path)  # create model from config

        if pretrained:
            # 1. Build model with correct nc
            model.nc = self.num_classes
            model.names = list(range(self.num_classes))

            print("🔁 Loading pretrained YOLOv8 weights (head will be skipped)...")

            ckpt = torch.load(weights_path, map_location="cpu")

            state_dict = (
                ckpt["model"].state_dict()
                if isinstance(ckpt, dict) and "model" in ckpt
                else ckpt
            )

            # 2. Filter incompatible head weights
            filtered = {}
            model_sd = model.state_dict()

            for k, v in state_dict.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    filtered[k] = v
                else:
                    if "cv3" in k:
                        print(f"Skipping head weight: {k}")
                    else:
                        print(f"Skipping {k}: {v.shape} vs {model_sd.get(k)}")

            missing, unexpected = model.load_state_dict(filtered, strict=False)

            print("Missing:", missing)
            print("Unexpected:", unexpected)
            print("✅ Pretrained YOLOv8 weights loaded.")
        else:
            print("⚠️ Initializing YOLOv8 model with random weights.")
        # return YOLOv8Wrapper(model, num_classes=7, device=DEVICE)
        return YOLOv8Wrapper(
            model,
            num_classes=self.num_classes,
            device=self.device,
        )

    def load_yolo_v5(
        self,
        repo_path: str = "ultralytics/yolov5",
        pretrained: bool = True,
        model_name: str = "yolov5s",
        model_dir: str = "models/yolov5",
        loss_yaml: str = "loss.yaml",
    ) -> YOLOv5Wrapper:

        if pretrained:
            wrapper = torch.hub.load(
                repo_path,
                "custom",
                autoshape=False,
                path=f"{model_dir}/{model_name}.pt",
            )
            model = wrapper.model
        else:
            model = torch.hub.load(
                repo_path,
                model_name,
                pretrained=False,
                autoshape=False,
                force_reload=True,
            )

        # load hyperparameters
        with open(Path(model_dir) / loss_yaml) as f:
            hyp = yaml.safe_load(f)
            model.hyp = hyp

        return YOLOv5Wrapper(
            model=model,
            num_classes=self.num_classes,
            device=self.device,
        )

    def load_retinanet_pipeline(
        self,
        pretrained=True,
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT,
    ) -> RetinaNetWrapper:

        model = retinanet_resnet50_fpn(
            weights=weights if pretrained else None,
        )

        return RetinaNetWrapper(
            model=model,
            num_classes=self.num_classes,
            device=self.device,
        )

    def load_detr_pipeline(
        self,
        pretrained=True,
        weights="facebook/detr-resnet-50",
    ) -> DetrWrapper:

        if pretrained:
            config = DetrConfig.from_pretrained(weights)
            config.num_labels = self.num_classes
            model = DetrForObjectDetection.from_pretrained(
                weights, config=config, ignore_mismatched_sizes=True
            )
        else:
            model = DetrForObjectDetection(
                config=DetrConfig(num_labels=self.num_classes)
            )
        return DetrWrapper(
            model=model,
            num_classes=self.num_classes,
            device=self.device,
        )
