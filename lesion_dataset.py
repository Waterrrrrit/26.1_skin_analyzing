import os
import json
from PIL import Image
from torch.utils.data import Dataset

# 이미지 기준 명칭으로 통일 (피지선결석 적용)
LESION_CLASSES_13 = [
    "화이트헤드", "블랙헤드", "구진", "농포", "결절", "피지선결석",
    "비립종", "한관종", "모공확장", "기미", "색소침착", "주사(딸기코)", "지루성 피부염"
]

class LesionClassificationDataset(Dataset):
    def __init__(self, image_dir: str, json_dir: str, transform=None):
        """
        AI-Hub 데이터 구조에 맞춘 커스텀 데이터셋
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(LESION_CLASSES_13)}
        
        valid_ext = ('.jpg', '.jpeg', '.png')
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(valid_ext)
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 전체 로직을 try-except로 감싸 에러 발생 시 스킵(None 반환) 처리
        try:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            
            json_name = os.path.splitext(img_name)[0] + '.json'
            json_path = os.path.join(self.json_dir, json_name)

            # 1. 이미지 로드
            image = Image.open(img_path).convert('RGB')

            # 2. JSON 파싱
            with open(json_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
                diagnosis = meta_data.get('diagnosis_name', '')

            if diagnosis not in self.class_to_idx:
                raise ValueError(f"정의되지 않은 클래스입니다: {diagnosis}")
                
            label = self.class_to_idx[diagnosis]

            # 3. 전처리 적용
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            # 에러 발생 시 로그만 남기고 None 반환 (DataLoader에서 필터링됨)
            print(f"\n[Warning] 데이터 로드 실패. 스킵합니다. (파일: {img_name}) | 사유: {e}")
            return None
