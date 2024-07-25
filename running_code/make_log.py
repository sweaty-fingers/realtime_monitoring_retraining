import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[1])))

import pandas as pd
import numpy as np
import cv2
import torch
import os
from PIL import Image
import shutil
import time
from retraining.models.torch.deeplabv3 import ImageSegmenationDeepLabV3
from retraining.models.torch.mobilenetv3 import ImageSegmenationMobilenetV3

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DATA_DIR = os.path.join(ROOT_DIR, "captured_images")
LOG_DIR = os.path.join(ROOT_DIR, "logs", "realtime_monitoring")

SUCCESS_TRRESHOLD = 0.98
TARGET_SIZE = (257, 257)

class RealTimeSegmentation():
    def __init__(self, device_model_path=None, base_model_path=None, comparison_steps=5, log_dir=None, root_data_dir=None):
        
        self.base_model = ImageSegmenationDeepLabV3(model_path=base_model_path)
        self.device_model = ImageSegmenationMobilenetV3(model_path=device_model_path)
        self.target_size = TARGET_SIZE

        self.root_data_dir = root_data_dir
        if self.root_data_dir is None:
            self.root_data_dir = ROOT_DATA_DIR
        self.old_data_dir = os.path.join(self.root_data_dir, "old")

        self.log_dir = log_dir
        if log_dir is None:
            self.log_dir = LOG_DIR

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.old_log_file = os.path.join(self.log_dir, "old_model_log.csv")
        self.comparison_steps = comparison_steps

        self.steps_above_threshold = 0
        self.success_count = 0
        self.fail_count = 0

        self.success_threshold = SUCCESS_TRRESHOLD
        self.colormap = ImageSegmenationMobilenetV3.get_colormap()

        self.setup_directories()

    def setup_directories(self):
        self.unseen_dir = os.path.join(self.old_data_dir, 'unseen')
        
        self.seen_success_origin_dir = os.path.join(self.old_data_dir, 'seen', 'success', 'origin')
        self.seen_fail_origin_dir = os.path.join(self.old_data_dir, 'seen', 'fail', 'origin')

        self.seen_success_pred_mask_dir = os.path.join(self.old_data_dir, 'seen', 'success', 'pred_mask')
        self.seen_success_true_mask_dir = os.path.join(self.old_data_dir, 'seen', 'success', 'true_mask')

        self.seen_fail_pred_mask_dir = os.path.join(self.old_data_dir, 'seen', 'fail', 'pred_mask')
        self.seen_fail_true_mask_dir = os.path.join(self.old_data_dir, 'seen', 'fail', 'true_mask')

        self.seen_pred_combined_dir = os.path.join(self.old_data_dir, 'seen', 'combined', 'pred')
        self.seen_true_combined_dir = os.path.join(self.old_data_dir, 'seen', 'combined', 'true')
        
        os.makedirs(self.seen_success_origin_dir, exist_ok=True)
        os.makedirs(self.seen_fail_origin_dir, exist_ok=True)

        os.makedirs(self.seen_success_pred_mask_dir, exist_ok=True)
        os.makedirs(self.seen_success_true_mask_dir, exist_ok=True)

        os.makedirs(self.seen_fail_pred_mask_dir, exist_ok=True)
        os.makedirs(self.seen_fail_true_mask_dir, exist_ok=True)

        os.makedirs(self.seen_pred_combined_dir, exist_ok=True)
        os.makedirs(self.seen_true_combined_dir, exist_ok=True)

    def create_color_mask(self, preds_mask):
        color_mask = np.zeros((preds_mask.shape[0], preds_mask.shape[1], 3), dtype=np.uint8)
        for class_id in range(21):
            color_mask[preds_mask == class_id] = self.colors[class_id]
        return color_mask
    
    # def apply_mask(self, frame, output):
    #     mask = (output * 255 / output.max()).astype(np.uint8)
    #     mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    #     mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    #     combined = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
    #     return combined, mask_colored
    
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
            for file in os.listdir(self.unseen_dir):
                if file.endswith(('jpg', 'jpeg', 'png')):
                    image = cv2.imread(os.path.join(self.unseen_dir, file))
                    images.append((os.path.join(self.unseen_dir, file), image))
        except Exception as e:
            # st.write(f"Failed to read images from directory: {e}, \n {self.unseen_dir}")
            return None
        return images

    def calculate_segmentation_accuracy(self, true_label_tensor, predicted_label_tensor):
        
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
            print(f"{self.unseen_dir} no data")
            return
        try:
            self.old_df = pd.read_csv(self.old_log_file)
        except FileNotFoundError:
            columns = ['step', 'success', 'fail', 'cumulative_success_rate', 'accuracy', 'original_image_path', 'true_mask_image_path', 'pred_mask_image_path', 'true_combined_image_path', 'pred_combined_image_path']
            self.old_df = pd.DataFrame(columns=columns)

        for img_path, image in images:
            # Preprocess image

            base_input_image = self.base_model.preprocess_image(image)
            device_input_image = self.device_model.preprocess_image(image)

            # Run model inference
            base_prediction = self.base_model.run_inference(base_input_image)
            device_prediction = self.device_model.run_inference(device_input_image)

            # Apply mask
            true_masked, true_colored_mask = self.apply_mask(image, base_prediction)
            pred_masked, pred_colored_mask = self.apply_mask(image, device_prediction)

            # For demonstration, determine success randomly
            accuracy = self.get_success(true=base_prediction, pred=device_prediction)
            if accuracy > self.success_threshold: # success
                self.success_count += 1
                success = 1
                fail = 0
                original_img_path = os.path.join(self.seen_success_origin_dir, os.path.basename(img_path))
                shutil.move(img_path, original_img_path)
                self.save_mask(pred_colored_mask.astype(np.uint8), os.path.join(self.seen_success_pred_mask_dir, os.path.basename(img_path).rsplit(".")[0] + ".png"))
                self.save_mask(true_colored_mask.astype(np.uint8), os.path.join(self.seen_success_true_mask_dir, os.path.basename(img_path).rsplit(".")[0] + ".png"))
                # self.save_mask(true_colored_mask.numpy().astype(np.uint8), os.path.join(self.seen_success_true_mask_dir, os.path.basename(img_path).rsplit(".")[0] + ".png"))
                # cv2.imwrite(os.path.join(self.seen_success_pred_mask_dir, os.path.basename(img_path)), device_prediction)
                # cv2.imwrite(os.path.join(self.seen_success_true_mask_dir, os.path.basename(img_path)), base_prediction)

            else:
                self.fail_count += 1
                success = 0
                fail = 1
                original_img_path = os.path.join(self.seen_fail_origin_dir, os.path.basename(img_path))
                shutil.move(img_path, original_img_path)
                self.save_mask(pred_colored_mask.astype(np.uint8), os.path.join(self.seen_fail_pred_mask_dir, os.path.basename(img_path).rsplit(".")[0] + ".png"))
                self.save_mask(true_colored_mask.astype(np.uint8), os.path.join(self.seen_fail_true_mask_dir, os.path.basename(img_path).rsplit(".")[0] + ".png"))
                # cv2.imwrite(os.path.join(self.seen_fail_pred_mask_dir, os.path.basename(img_path)), device_prediction)
                # cv2.imwrite(os.path.join(self.seen_fail_true_mask_dir, os.path.basename(img_path)), base_prediction)

            cv2.imwrite(os.path.join(self.seen_pred_combined_dir, os.path.basename(img_path)), pred_masked)
            cv2.imwrite(os.path.join(self.seen_true_combined_dir, os.path.basename(img_path)), true_masked)

            cumulative_success_rate = self.success_count / (self.success_count + self.fail_count)
            
            new_row_old = {
                'step': len(self.old_df) + 1,
                'success': success,
                'fail': fail,
                'cumulative_success_rate': cumulative_success_rate,
                'accuracy': accuracy,
                'original_image_path': original_img_path,
                'true_mask_image_path': os.path.join(self.seen_success_true_mask_dir if success else self.seen_fail_true_mask_dir, os.path.basename(img_path).rsplit(".")[0] + ".png"),
                'pred_mask_image_path': os.path.join(self.seen_success_pred_mask_dir if success else self.seen_fail_pred_mask_dir, os.path.basename(img_path).rsplit(".")[0] + ".png"),
                'true_combined_image_path': os.path.join(self.seen_true_combined_dir, os.path.basename(img_path)),
                'pred_combined_image_path': os.path.join(self.seen_pred_combined_dir, os.path.basename(img_path))
            }
            
            self.old_df = pd.concat([self.old_df, pd.DataFrame([new_row_old])], ignore_index=True)

            # Save updated data
            self.save_data()
    
    def save_mask(self, mask, path):
        mask_image = Image.fromarray(mask)
        mask_image.save(path)

    def save_data(self):
        self.old_df.to_csv(self.old_log_file, index=False)

if __name__ == '__main__':

    try:
        seg = RealTimeSegmentation()

        while True:
            seg.update_data()    
            time.sleep(1)

    except KeyboardInterrupt:
        print("logging stopped.")
    
    # main()
    # st.write("Real-time segmentation accuracy comparison for TFLite model.")
