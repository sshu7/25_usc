import cv2
import torch
import numpy as np
from torchvision import transforms

# 분류기 입력 전처리(학습과 동일해야 함)
def build_transform(img_size=160):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def crop_safe(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

