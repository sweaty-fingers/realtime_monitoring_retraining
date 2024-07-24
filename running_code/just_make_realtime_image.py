import cv2
import time
import os, sys
from pathlib import Path
import shutil

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 저장할 디렉토리 설정
save_dir = os.path.join(base_dir, 'captured_images')
latest_dir = os.path.join(save_dir, 'latest')
seen_dir = os.path.join(save_dir, 'old', 'unseen')

# 디렉토리 생성
os.makedirs(save_dir, exist_ok=True)
os.makedirs(latest_dir, exist_ok=True)
os.makedirs(seen_dir, exist_ok=True)

# 카메라 캡처 객체 생성 (기본 카메라: 0번)
cap = cv2.VideoCapture(0)

# 캡처 간격 (초)
capture_interval = 1
# 최신 이미지를 유지할 개수
latest_count = 20

def manage_images():
    # latest_dir에 있는 모든 이미지 파일 목록 가져오기
    images = [f for f in os.listdir(latest_dir) if os.path.isfile(os.path.join(latest_dir, f))]
    images.sort()  # 파일명으로 정렬 (시간순)

    # 최신 이미지를 초과한 파일을 seen_dir로 이동
    while len(images) > latest_count:
        image_to_move = images.pop(0)
        shutil.move(os.path.join(latest_dir, image_to_move), os.path.join(seen_dir, image_to_move))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # 현재 시간으로 파일 이름 생성
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"image_{timestamp}.jpg"
        file_path = os.path.join(latest_dir, file_name)
        
        # 이미지 저장
        cv2.imwrite(file_path, frame)
        print(f"Saved {file_path}")

        # 이미지 관리 함수 호출
        manage_images()

        # 캡처 간격만큼 대기
        time.sleep(capture_interval)

except KeyboardInterrupt:
    print("Image capturing stopped.")

finally:
    # 캡처 객체 해제
    cap.release()
    cv2.destroyAllWindows()
