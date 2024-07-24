import os
from pathlib import Path

import torch
import torchvision.models as models
from torchvision import transforms

ROOT_DIR = str((Path(os.path.abspath(__file__)).parents[3]))
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "deeplabv3.pth")

class ImageSegmenationDeepLabV3():
    def __init__(self, model_path=None, num_classes=21, input_shape=(256, 256),):
        if model_path is None:
            model_path = MODEL_PATH

        self.model = ImageSegmenationDeepLabV3.load_model(model_path=model_path)
        
        self.model.eval()
        self.input_shape = input_shape

    @staticmethod
    def load_model(model_path=None):
        model = models.segmentation.deeplabv3_resnet101(pretrained=False)
        model.load_state_dict(torch.load(model_path), strict=False)
        return model    
    
    def preprocess_image(self, image):
        preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        return input_batch
    
    def run_inference(self, input_batch):
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
            output_predictions = output.argmax(0)
        
        return output_predictions