import os
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

ROOT_DIR = str((Path(os.path.abspath(__file__)).parents[3]))
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "mobilenetv3.pth")
INPUT_SHAPE = (256, 256)

class ImageSegmenationMobilenetV3():
    def __init__(self, model_path=None, num_classes=21, input_shape=None,):
        if model_path is None:
            model_path = MODEL_PATH

        self.model = ImageSegmenationMobilenetV3.load_model(model_path=model_path, num_classes=num_classes)
        
        self.model.eval()
        self.input_shape = input_shape
        if self.input_shape is None:
            self.input_shape = INPUT_SHAPE

        self.color_map = ImageSegmenationMobilenetV3.get_colormap()
        
    @staticmethod
    def load_model(model_path=None, num_classes=21):
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False)
    
        # Custom number of classes (default is 21 for COCO dataset)
        if num_classes != 21:
            model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

        model.load_state_dict(torch.load(model_path), strict=False)
        return model
    
    def preprocess_image(self, image):
        preprocess = ImageSegmenationMobilenetV3.get_preprocess()
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        return input_batch
    
    @staticmethod
    def get_colormap():
        colormap = np.array([
            [0, 0, 0],        # Background
            [128, 0, 0],      # Aeroplane
            [0, 128, 0],      # Bicycle 
            [128, 128, 0],    # Bird
            [0, 0, 128],      # Boat
            [128, 0, 128],    # Bottle
            [0, 128, 128],    # Bus
            [128, 128, 128],  # Car
            [64, 0, 0],       # Cat
            [192, 0, 0],      # Chair
            [64, 128, 0],     # Cow
            [192, 128, 0],    # Dining table
            [64, 0, 128],     # Dog
            [192, 0, 128],    # Horse
            [64, 128, 128],   # Motorbike
            [192, 128, 128],  # Person
            [0, 64, 0],       # Potted plant
            [128, 64, 0],     # Sheep
            [0, 192, 0],      # Sofa
            [128, 64, 128],   # Train
            [0, 192, 128],    # TV/monitor
        ], dtype=np.uint8)

        return colormap

    @staticmethod
    def get_preprocess():
        preprocess = transforms.Compose([                    
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        return preprocess

    @staticmethod
    def get_transform(input_shape):
        preprocess = transforms.Compose([                    
                    transforms.ToTensor(),
                    transforms.Resize(input_shape),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        return preprocess

    @staticmethod
    def get_mask_transform(input_shape):
        preprocess = transforms.Compose([                    
                    transforms.Resize(input_shape, interpolation=Image.NEAREST)
                ])
        
        return preprocess


    def run_inference(self, input_batch):
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
            output_predictions = output.argmax(0)
        
        return output_predictions