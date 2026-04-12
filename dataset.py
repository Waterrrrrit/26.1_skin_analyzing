import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from PIL import Image
from torch.utils.data import Dataset

from Preprocessing_reinhard import apply_reinhard_norm

@dataclass(frozen=True)
class LesionSample:
    image_path: Path
    json_path: Path
    label_name: str
    class_index: int

class AdvancedLesionDataset(Dataset):
    SAMPLE_FOLDER_KEYWORD = "샘플"
    
    LESION_CLASSES = [
        'Whitehead', 'Blackhead', 'Papule', 'Pustule', 'Nodule', 
        'Sebaceous_Calculi', 'Milia', 'Syringoma', 'Enlarged_Pores', 
        'Melasma', 'PIH', 'Rosacea', 'Seborrheic_Dermatitis',
        '건선', '아토피', '여드름', '정상', '주사', '지루'
    ]
    
    LABEL_TO_INDEX = {
        label: (idx if idx < 13 else (idx % 13)) 
        for idx, label in enumerate(LESION_CLASSES)
    }

    def __init__(self, dataset_root, split, target_stats, transform=None):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.target_stats = target_stats
        self.transform = transform
        
        self.image_root = self.dataset_root / split / "01.원천데이터"
        self.label_root = self.dataset_root / split / "02.라벨링데이터"
        
        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_array = np.fromfile(str(sample.image_path), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        image = apply_reinhard_norm(image, self.target_stats)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(sample.class_index, dtype=torch.long)

    def _build_samples(self) -> List[LesionSample]:
        samples = []
        label_index = {p.stem: p for p in self.label_root.rglob("*.json")}
        
        for image_dir in sorted(self.image_root.iterdir()):
            if not image_dir.is_dir() or self.SAMPLE_FOLDER_KEYWORD in image_dir.name:
                continue
                
            for ext in ["*.png", "*.jpg"]:
                for image_path in image_dir.glob(ext):
                    json_path = label_index.get(image_path.stem)
                    if not json_path:
                        continue
                    
                    label_name = self._parse_json(json_path)
                    if label_name:
                        found_cls = next((c for c in self.LESION_CLASSES if c in label_name or c in image_dir.name), None)
                        if found_cls:
                            samples.append(LesionSample(
                                image_path, 
                                json_path, 
                                found_cls, 
                                self.LABEL_TO_INDEX[found_cls]
                            ))
        return samples

    def _parse_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["annotations"][0]["diagnosis_info"]["diagnosis_name"]
        except Exception:
            return None