import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[2])))

import csv

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from retraining.data import CustomDataset, VOCSubset

from retraining.models.torch.mobilenetv3 import ImageSegmenationMobilenetV3
from retraining.models.torch.deeplabv3 import ImageSegmenationDeepLabV3

ROOT_DIR = str((Path(os.path.abspath(__file__)).parents[2]))
LOG_DIR = os.path.join(ROOT_DIR, "logs", "training_loss")

ST_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "mobilenetv3.pth")
TE_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "deeplabv3.pth")
SAVE_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "torch", "mobilenetv3_tmp.pth")
COMPARING_SCRIPT_PATH = os.path.join(ROOT_DIR, "running_code", "compare_performance.py")

MAX_EPOCH = 5
CUSTOM_RATIO = 0.4
VOC_RATIO = 1 - CUSTOM_RATIO
BATCH_SIZE = 4

def distillation_loss(student_outputs, teacher_outputs, targets, alpha=0.5, temperature=3.0):
    """
    지식 증류 손실 함수
    student_outputs: 학생 모델의 예측
    teacher_outputs: 교사 모델의 예측
    targets: 실제 레이블
    alpha: CE loss와 distillation loss의 비율
    temperature: 온도 파라미터
    """
    
    ce_loss = F.cross_entropy(student_outputs, targets)

    # Logits를 온도로 나누고 softmax 적용
    student_soft = F.log_softmax(student_outputs / temperature, dim=1)
    teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)

    # Distillation loss
    distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * distillation_loss

# 옵티마이저 설정
def save_model(model, epoch, optimizer, path="model.pth"):
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, path)
    torch.save(model.state_dict(), path)

def training(teacher_model, student_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    # 데이터 전처리
    custom_dataset = CustomDataset()
    voc_dataset = VOCSubset()

    combined_dataset = ConcatDataset([voc_dataset, custom_dataset])
    # 각 데이터셋의 비율 설정
    voc_ratio = 0.6
    custom_ratio = 0.4

    # 각 데이터셋의 샘플링 비율 계산
    num_voc = len(voc_dataset)
    num_custom = len(custom_dataset)
    total_samples = num_voc + num_custom
    weights = [voc_ratio / num_voc] * num_voc + [custom_ratio / num_custom] * num_custom

    sampler = WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # 학습 루프
    # CSV 파일 초기화
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

    csv_file = os.path.join(LOG_DIR, 'training_loss.csv')
    with open(csv_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])

    num_epochs = MAX_EPOCH
    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Teacher 모델 예측 (gradient 필요 없음)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)['out']
            
            # Student 모델 예측
            student_outputs = student_model(images)['out']
            
            # 손실 계산
            loss = distillation_loss(student_outputs, teacher_outputs, targets)
            epoch_loss += loss.item()

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

            # CSV 파일에 기록
        with open(csv_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss])
        
        save_model(model=student_model, epoch=epoch + 1, optimizer=optimizer, path=SAVE_MODEL_PATH)

if __name__ == "__main__":
    teacher_model = ImageSegmenationDeepLabV3.load_model(TE_MODEL_PATH)
    student_model = ImageSegmenationMobilenetV3.load_model(ST_MODEL_PATH)
    
    training(teacher_model=teacher_model, student_model=student_model)
    os.system(f"python {COMPARING_SCRIPT_PATH}")
