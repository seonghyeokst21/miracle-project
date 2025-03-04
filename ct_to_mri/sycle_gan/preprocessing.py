import cv2
import numpy as np
import os
from glob import glob

def resize_and_pad(image, target_size=256):
    """이미지를 비율을 유지하면서 중앙에 배치하고, 256x256 크기로 패딩"""
    h, w = image.shape[:2]
    print(f"원본 이미지 크기: {h}x{w}")  # 원본 이미지 크기 출력
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    print(f"조정된 크기: {new_h}x{new_w}")  # 리사이징 후 크기 출력
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print("리사이즈 완료")
    
    padded = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    print(f"패딩 (top, left): ({pad_top}, {pad_left})")  # 패딩 크기 출력
    
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    print("패딩 완료")
    
    return padded

def process_images(input_folder, output_folder):
    """폴더 내 모든 이미지를 256x256 크기로 변환"""
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob(os.path.join(input_folder, "*.png"))  # 확장자는 필요에 맞게 변경
    print(f"이미지 경로 목록: {image_paths}")  # 이미지 경로 리스트 출력
    
    if not image_paths:
        print("경로에 이미지가 없습니다.")
        return
    
    for img_path in image_paths:
        print(f"처리 중 이미지: {img_path}")  # 현재 처리 중인 이미지 경로 출력
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # CT 이미지는 보통 흑백
        
        if image is None:
            print(f"이미지 읽기 실패: {img_path}")  # 이미지 읽기 실패한 경우 출력
            continue
        
        resized_image = resize_and_pad(image)
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, resized_image)
        print(f"변환 완료: {output_path}")  # 변환 완료된 이미지 경로 출력
    

# 사용 예시
input_folder = "C:/Users/seong/Code/ct_to_mri/Dataset/images/testA"
output_folder = "C:/Users/seong/Code/ct_to_mri/Dataset_processed/resize_and_padding/testA"
print("실행1")
process_images(input_folder, output_folder)
