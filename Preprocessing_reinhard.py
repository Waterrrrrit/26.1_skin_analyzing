import cv2
import numpy as np

def apply_reinhard_norm(image, target_stats):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
    l, a, b = cv2.split(lab)

    l_m, l_s = l.mean(), l.std()
    a_m, a_s = a.mean(), a.std()
    b_m, b_s = b.mean(), b.std()

    eps = 1e-5
    l = (l - l_m) * (target_stats['l_std'] / (l_s + eps)) + target_stats['l_mean']
    a = (a - a_m) * (target_stats['a_std'] / (a_s + eps)) + target_stats['a_mean']
    b = (b - b_m) * (target_stats['b_std'] / (b_s + eps)) + target_stats['b_mean']

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    merged = cv2.merge([l, a, b]).astype("uint8")
    
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)