#pip install ultralytics

from ultralytics import YOLO

# 모델 학습 용 데이터셋 설정 파일 생성
# RDD2022 5개 클래스 정의
yaml_content = """
path: /content/RDD2022_Data/RDD_SPLIT
train: train/images
val: val/images
nc: 5
names: ['longitudinal crack', 'transverse crack', 'alligator crack', 'other corruption', 'Pothole']
"""
with open('/content/RDD2022_Data/my_data.yaml', 'w') as f:
    f.write(yaml_content.strip())

print("my_data.yaml 생성완료")


# yolov8s 모델로 기초 학습

model = YOLO('yolov8s.pt')

results = model.train(
    data='/content/RDD2022_Data/my_data.yaml',
    epochs=120,
    imgsz=640,
    batch=32,
    workers=8,
    project='/content/drive/MyDrive/Pothole_Project',
    name='scratch_training'
)

# 1차로 학습한 가중치를 불러와 데이터 증강 적용
model = YOLO('/content/drive/MyDrive/Pothole_Project/scratch_training/weights/best.pt')
results = model.train(
    data='/content/RDD2022_Data/my_data.yaml',
    epochs=60,
    imgsz=640,
    batch=32,
    workers=8,
    device=0, # gpu 사용
    project='/content/drive/MyDrive/Pothole_Project',
    name='augmented_training',
    hsv_v=0.6, # 명도 변화
    hsv_s=0.5, # 채도 변화
    perspective=0.001, # 원근 왜곡
    scale=0.6, # 크기 스케일링
    erasing=0.0, # 랜덤 지우기 미사용
    mosaic=0.5 # 모자이크 기법 비율
)