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


class ModelFactory:
    """
    Factory class for loading and initializing various object detection models.

    This factory provides methods to load different state-of-the-art object detection
    models including YOLOv8, YOLOv5, RetinaNet, and DETR. Each model is wrapped with
    a corresponding wrapper class that provides a unified interface.

    Attributes:
        device (str): The device to load models on ('cuda' or 'cpu').
        num_classes (int): Number of object classes for detection tasks.

    Methods:
        load_yolo_v8: Load a YOLOv8 model with optional pretrained weights.
        load_yolo_v5: Load a YOLOv5 model from torch hub with optional pretrained weights.
        load_retinanet_pipeline: Load a RetinaNet model with ResNet50 FPN backbone and optional pretrained weights.
        load_detr_pipeline: Load a DETR (Detection Transformer) model from Hugging Face with optional pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.num_classes = num_classes

        print(f"Model factory initialized. Using device: {self.device}")

    def load_yolo_v8(
        self,
        pretrained: bool = True,
        model_name: str = "yolov8s",
        model_dir: str = "models/yolov8",
    ) -> YOLOv8Wrapper:

        weights_path = Path(model_dir) / f"{model_name}.pt"

        if pretrained:
            wrapper = YOLO(str(weights_path))
            model = wrapper.model

        else:
            yaml_path = Path(model_dir) / f"{model_name}.yml"

            print("⚠ Loading YOLOv8 model without pretrained weights...")

            # Create model from config
            model = DetectionModel(yaml_path)
            ckpt = torch.load(weights_path, map_location="cpu")

            if "model" in ckpt:
                state_dict = (
                    ckpt["model"].state_dict()
                    if hasattr(ckpt["model"], "state_dict")
                    else ckpt["model"]
                )
            else:
                print("is dict")
                state_dict = ckpt  # raw state_dict

            # filter fitting weights
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(
                        f"Skipping {k}, shape mismatch: {v.shape} vs {model.state_dict()[k].shape}"
                    )

            # Load filtered weights
            missing, unexpected = model.load_state_dict(filtered_dict, strict=False)

            print("Missing: ", missing)
            print("Unexpected: ", unexpected)

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
        weight_path: str = "models/yolov5",
        loss_yaml: str = "loss.yaml",
    ) -> YOLOv5Wrapper:

        if pretrained:
            wrapper = torch.hub.load(
                repo_path,
                "custom",
                autoshape=False,
                path=f"{weight_path}/{model_name}.pt",
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
        with open(Path(weight_path) / loss_yaml) as f:
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
