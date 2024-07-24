import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[1])))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import cv2
import torch
import os
from PIL import Image
import shutil
import time
from retraining.models.torch.deeplabv3 import ImageSegmenationDeepLabV3
from retraining.models.torch.mobilenetv3 import ImageSegmenationMobilenetV3

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "logs", "comparing_ab_model")
TEST_DATA_DIR = os.path.join(ROOT_DIR, "captured_images", "latest")

OLD_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "mobilenetv3.pth")
NEW_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "mobilenetv3_tmp.pth")

ALERT_THRESHOLD = 0.45
SUCCESS_TRRESHOLD = 0.995
TARGET_SIZE = (257, 257)

class RealTimeCompare():
    def __init__(self, device_model_path=None, base_model_path=None, comparison_steps=5, log_dir=None, root_data_dir=None):
        self.old_model_path = OLD_MODEL_PATH
        self.new_model_path = NEW_MODEL_PATH

        self.base_model = ImageSegmenationDeepLabV3(model_path=base_model_path)
        self.device_model = ImageSegmenationMobilenetV3(model_path=self.old_model_path)
        self.new_model = ImageSegmenationMobilenetV3(model_path=self.new_model_path)

        self.target_size = TARGET_SIZE
        self.test_data_dir = TEST_DATA_DIR
        
        self.log_dir = LOG_DIR
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, "compaing_result.csv")
        
        self.success_count = 0
        self.continual_success_count = 0
        self.fail_count = 0

        self.colormap = ImageSegmenationMobilenetV3.get_colormap()

    def create_color_mask(self, preds_mask):
        color_mask = np.zeros((preds_mask.shape[0], preds_mask.shape[1], 3), dtype=np.uint8)
        for class_id in range(21):
            color_mask[preds_mask == class_id] = self.colors[class_id]
        return color_mask
    
    def apply_mask(self, image, mask, alpha=0.5):
        # Convert PIL image to numpy array
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        # Convert mask to numpy array and color it
        mask = mask.cpu().numpy()
        color_mask = self.colormap[mask]

        # Resize color mask to match the image size
        color_mask_resized = cv2.resize(color_mask.astype(np.float32), (width, height))
        
        # Blend the image and the color mask
        overlay = (1 - alpha) * image_array + alpha * color_mask_resized
        
        return overlay, color_mask

    def get_data(self):
        images = []
        try:
            for file in os.listdir(self.test_data_dir):
                if file.endswith(('jpg', 'jpeg', 'png')):
                    image = cv2.imread(os.path.join(self.test_data_dir, file))
                    images.append((os.path.join(self.test_data_dir, file), image))
        except Exception as e:
            st.write(f"Failed to read images from directory: {e}, \n {self.test_data_dir}")
            return None
        return images

    def calculate_segmentation_accuracy(self, true_label_tensor, predicted_label_tensor):
        # numpy 배열을 torch 텐서로 변환
        # true_label_tensor = torch.tensor(true_label_array)
        # predicted_label_tensor = torch.tensor(predicted_label_array)
        
        # true_label_tensor와 predicted_label_tensor의 형태 확인 (H, W)
        if true_label_tensor.shape != predicted_label_tensor.shape:
            raise ValueError("true_label_array and predicted_label_array must have the same shape")

        # true_label_tensor와 predicted_label_tensor의 dtype이 정수형인지 확인
        true_label_tensor = true_label_tensor.long()
        predicted_label_tensor = predicted_label_tensor.long()

        # 픽셀 단위 비교
        true_positive = torch.sum((true_label_tensor == predicted_label_tensor) & (true_label_tensor != 0))
        true_negative = torch.sum((true_label_tensor == 0) & (predicted_label_tensor == 0))
        false_positive = torch.sum((true_label_tensor == 0) & (predicted_label_tensor != 0))
        false_negative = torch.sum((true_label_tensor != 0) & (predicted_label_tensor == 0))

        # 정확도 계산
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        
        # 정확도 반환
        return accuracy.item()
    
    def get_success(self, true, pred):
        return self.calculate_segmentation_accuracy(true, pred)

        # return np.random.choice([True, False])

    def update_data(self):
        images = self.get_data()
        if images is None:
            print(f"{self.test_data_dir} no data")
            return
        try:
            self.old_df = pd.read_csv(self.log_path)
        except FileNotFoundError:
            columns = ['step', 'success', 'fail', 'cumulative_success_rate', 'continual_success_count', 'old_accuracy' ,'new_accuracy']
            self.old_df = pd.DataFrame(columns=columns)

        for img_path, image in images:
            # Preprocess image

            base_input_image = self.base_model.preprocess_image(image)
            device_input_image = self.device_model.preprocess_image(image)
            new_input_image = self.new_model.preprocess_image(image)

            # Run model inference
            base_prediction = self.base_model.run_inference(base_input_image)
            device_prediction = self.device_model.run_inference(device_input_image)
            new_prediction = self.new_model.run_inference(new_input_image)

            # For demonstration, determine success randomly
            old_accuracy = self.get_success(true=base_prediction, pred=device_prediction)
            new_accuracy = self.get_success(true=base_prediction, pred=new_prediction)

            if new_accuracy > old_accuracy: # success
                self.success_count += 1
                self.continual_success_count += 1
                success = 1
                fail = 0

            else:
                self.fail_count += 1
                self.continual_success_count = 0
                success = 0
                fail = 1

            cumulative_success_rate = self.success_count / (self.success_count + self.fail_count)
            
            new_row_old = {
                'step': len(self.old_df) + 1,
                'success': success,
                'fail': fail,
                'cumulative_success_rate': cumulative_success_rate,
                'continual_success_count': self.continual_success_count,
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
            }
            
            self.old_df = pd.concat([self.old_df, pd.DataFrame([new_row_old])], ignore_index=True)

            # Save updated data
            self.save_data()
    
    def save_mask(self, mask, path):
        mask_image = Image.fromarray(mask)
        mask_image.save(path)

    def save_data(self):
        self.old_df.to_csv(self.log_path, index=False)

if __name__ == '__main__':

    try:
        seg = RealTimeCompare()

        while True:
            seg.update_data()    
            time.sleep(1)

    except KeyboardInterrupt:
        print("logging stopped.")
    
    # main()
    # st.write("Real-time segmentation accuracy comparison for TFLite model.")
