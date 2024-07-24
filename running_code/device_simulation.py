import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[1])))

import cv2
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import shutil
import tkinter as tk
from threading import Thread

from retraining.models.torch.mobilenetv3 import ImageSegmenationMobilenetV3

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 저장할 디렉토리 설정
save_dir = os.path.join(base_dir, 'captured_images')
latest_dir = os.path.join(save_dir, 'latest')
seen_dir = os.path.join(save_dir, 'old', 'unseen')
# 디렉토리 생성
os.makedirs(save_dir, exist_ok=True)
os.makedirs(latest_dir, exist_ok=True)
os.makedirs(seen_dir, exist_ok=True)

# 캡처 간격 (초)
capture_interval = 2
# 최신 이미지를 유지할 개수
latest_count = 20

# 필터 적용할 타켓 클래스
TARGET_CLASSES = [15, 12, 13]
# 초기 필터 상태
current_filter = 'Normal'

# 이미지 저장 관리 매서드
def manage_images():
    # latest_dir에 있는 모든 이미지 파일 목록 가져오기
    images = [f for f in os.listdir(latest_dir) if os.path.isfile(os.path.join(latest_dir, f))]
    images.sort()  # 파일명으로 정렬 (시간순)

    # 최신 이미지를 초과한 파일을 seen_dir로 이동
    while len(images) > latest_count:
        image_to_move = images.pop(0)
        shutil.move(os.path.join(latest_dir, image_to_move), os.path.join(seen_dir, image_to_move))

# 정의한 색각이상 필터 함수들
def linearRGB_from_sRGB(v):
    fv = v / 255.0
    if fv < 0.04045:
        return fv / 12.92
    return ((fv + 0.055) / 1.055) ** 2.4

def sRGB_from_linearRGB(v):
    if v <= 0.:
        return 0
    if v >= 1.:
        return 255
    if v < 0.0031308:
        return 0.5 + (v * 12.92 * 255)
    return 255 * ((v ** (1.0 / 2.4)) * 1.055 - 0.055)

sRGB_to_linearRGB_Lookup = [linearRGB_from_sRGB(i) for i in range(256)]

brettel_params = {
    'protan': {
        'rgbCvdFromRgb_1': [
            0.14510, 1.20165, -0.34675,
            0.10447, 0.85316, 0.04237,
            0.00429, -0.00603, 1.00174
        ],
        'rgbCvdFromRgb_2': [
            0.14115, 1.16782, -0.30897,
            0.10495, 0.85730, 0.03776,
            0.00431, -0.00586, 1.00155
        ],
        'separationPlaneNormal': [0.00048, 0.00416, -0.00464]
    },
    'deutan': {
        'rgbCvdFromRgb_1': [
            0.36198, 0.86755, -0.22953,
            0.26099, 0.64512, 0.09389,
            -0.01975, 0.02686, 0.99289,
        ],
        'rgbCvdFromRgb_2': [
            0.37009, 0.88540, -0.25549,
            0.25767, 0.63782, 0.10451,
            -0.01950, 0.02741, 0.99209,
        ],
        'separationPlaneNormal': [-0.00293, -0.00645, 0.00938]
    },
    'tritan': {
        'rgbCvdFromRgb_1': [
            1.01354, 0.14268, -0.15622,
            -0.01181, 0.87561, 0.13619,
            0.07707, 0.81208, 0.11085,
        ],
        'rgbCvdFromRgb_2': [
            0.93337, 0.19999, -0.13336,
            0.05809, 0.82565, 0.11626,
            -0.37923, 1.13825, 0.24098,
        ],
        'separationPlaneNormal': [0.03960, -0.02831, -0.01129]
    }
}

def brettel(srgb, t, severity):
    rgb = np.array([sRGB_to_linearRGB_Lookup[val] for val in srgb])

    params = brettel_params[t]
    separationPlaneNormal = np.array(params['separationPlaneNormal'])
    rgbCvdFromRgb_1 = np.array(params['rgbCvdFromRgb_1']).reshape((3, 3))
    rgbCvdFromRgb_2 = np.array(params['rgbCvdFromRgb_2']).reshape((3, 3))

    dotWithSepPlane = np.dot(rgb, separationPlaneNormal)
    rgbCvdFromRgb = rgbCvdFromRgb_1 if dotWithSepPlane >= 0 else rgbCvdFromRgb_2

    rgb_cvd = np.dot(rgbCvdFromRgb, rgb)

    rgb_cvd = rgb_cvd * severity + rgb * (1.0 - severity)

    return np.array([sRGB_from_linearRGB(c) for c in rgb_cvd])

def monochrome_with_severity(srgb, severity):
    z = np.round(srgb[0] * 0.299 + srgb[1] * 0.587 + srgb[2] * 0.114)
    r = z * severity + (1.0 - severity) * srgb[0]
    g = z * severity + (1.0 - severity) * srgb[1]
    b = z * severity + (1.0 - severity) * srgb[2]
    return np.array([r, g, b])

brettel_functions = {
    'Normal': lambda v: v,
    'Protanopia': lambda v: brettel(v, 'protan', 1.0),
    'Protanomaly': lambda v: brettel(v, 'protan', 0.6),
    'Deuteranopia': lambda v: brettel(v, 'deutan', 1.0),
    'Deuteranomaly': lambda v: brettel(v, 'deutan', 0.6),
    'Tritanopia': lambda v: brettel(v, 'tritan', 1.0),
    'Tritanomaly': lambda v: brettel(v, 'tritan', 0.6),
    'Achromatopsia': lambda v: monochrome_with_severity(v, 1.0),
    'Achromatomaly': lambda v: monochrome_with_severity(v, 0.6)
}

def apply_filter(pixels, filter_name):
    filtered_pixels = np.array([brettel_functions[filter_name](pixel) for pixel in pixels])
    return filtered_pixels

# 필터 변경을 위한 Tkinter GUI 설정
def set_filter(filter_name):
    global current_filter
    current_filter = filter_name

def create_gui():
    root = tk.Tk()
    root.title("Color Blindness Filter")

    filters = ['Normal', 'Protanopia', 'Protanomaly', 'Deuteranopia', 'Deuteranomaly', 'Tritanopia', 'Tritanomaly', 'Achromatopsia', 'Achromatomaly']

    for filter_name in filters:
        button = tk.Button(root, text=filter_name, command=lambda name=filter_name: set_filter(name))
        button.pack()

    root.mainloop()

def get_target_mask(pred, target_classes):
    mask = np.zeros(pred.shape, dtype=np.uint8)
    for target_class in target_classes:
        mask[pred == target_class] = 1

    return mask

def main():
    
    try:
        cap = cv2.VideoCapture(0)
        # Tkinter를 메인 스레드에서 실행
        gui_thread = Thread(target=create_gui)
        gui_thread.start()

        # 모델 로드
        model = ImageSegmenationMobilenetV3()
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            input_image = model.preprocess_image(frame)
            with torch.no_grad():
                pred = model.run_inference(input_image)

            mask = get_target_mask(pred, TARGET_CLASSES)
            # 필터 적용할 영역 추출 및 필터 적용
            filtered_frame = frame.copy()
            mask_indices = np.where(mask == 1)
            if len(mask_indices[0]) > 0:
                filtered_pixels = apply_filter(frame[mask_indices], current_filter)
                filtered_frame[mask_indices] = filtered_pixels

            # 필터 적용된 프레임을 보여줌
            cv2.imshow('Filtered Video', filtered_frame)

            if (time.time() - start_time) > capture_interval:
                
                # 현재 시간으로 파일 이름 생성
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_name = f"image_{timestamp}.jpg"
                file_path = os.path.join(latest_dir, file_name)
                # 이미지 저장
                cv2.imwrite(file_path, frame)
                print(f"Saved {file_path}")
                # 이미지 관리 함수 호출
                manage_images()
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Image capturing stopped.")

    finally:
        # 캡처 객체 해제
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()