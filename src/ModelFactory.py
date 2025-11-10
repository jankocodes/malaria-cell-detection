import torch
# Import all necessary libraries for the models
from ultralytics import YOLO # For YOLOv8
from torchvision.models.detection import retinanet_resnet50_fpn
# import yolov5 # For YOLOv5 (if using PyTorch Hub or custom repo)
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from pathlib import Path



class ModelLoader:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Model factory initialized. Using device: {self.device}")
    
    
    def load_yolo_v8(self, pretrained=False, model_name="yolov8s", model_dir="models/yolov8"):
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
    