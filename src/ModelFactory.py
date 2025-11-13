import torch
# Import all necessary libraries for the models
from ultralytics import YOLO # For YOLOv8
from torchvision.models.detection import retinanet_resnet50_fpn
# import yolov5 # For YOLOv5 (if using PyTorch Hub or custom repo)
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection 
import torch



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
    
    def load_yolo_v5(self, 
                     repo_path: str = "models/yolov5",
                     pretrained: bool = False,
                     weights_path= None) -> str:
        
        if pretrained:
                model = torch.hub.load(
                repo_or_dir= repo_path, # the YOLOv5 repo
                model="custom", # tells it to use a local weights file
                source= "local",
                path=str(repo_path)+"/yolov5s.pt",
                autoshape= False
    )
        else: 
            model = torch.hub.load(repo_or_dir= repo_path,
                               model="yolov5s",
                               source="local",
                               autoshape=False, pretrained=False) # load scratch


        return model
        
    def load_retinanet_pipeline(self, 
                                pretrained=True,
                                weights= RetinaNet_ResNet50_FPN_Weights.DEFAULT):
        
        preprocess = weights.transforms() 
        
        model = retinanet_resnet50_fpn(
            weights=weights if pretrained else None, 
            score_thresh=0.7 
        )
        
        return (preprocess, model)
    
    def load_detr_pipeline(self, pretrained=True):
        
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50") if pretrained else DetrImageProcessor()
        
        if pretrained:
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        else:
            model = DetrForObjectDetection(DetrForObjectDetection.config_class())
        
        return(processor, model) 
  