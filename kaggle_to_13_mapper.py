import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from Preprocessing_reinhard import apply_reinhard_norm

class KaggleTo13Mapper(Dataset):
    def __init__(self, root_dir, target_stats, transform=None):
        self.root_dir = Path(root_dir)
        self.target_stats = target_stats
        self.transform = transform
        
        self.level_to_13_idx = {
            'Level_0': 0, 
            'Level_1': 1, 
            'Level_2': 3, 
            'Level_3': 4
        }
        
        self.samples = [
            (p, p.parent.name) 
            for p in self.root_dir.rglob("*.jpg") 
            if p.parent.name in self.level_to_13_idx
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, level = self.samples[idx]
        
        img_array = np.fromfile(str(path), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        image = apply_reinhard_norm(image, self.target_stats)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(self.level_to_13_idx[level], dtype=torch.long)