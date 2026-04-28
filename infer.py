import argparse
import json
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

# 사용자 정의 모듈 임포트
from model import build_efficientnet_b3
from preprocess import ReinhardNormalizer

def load_class_names(class_names_path: Path) -> list:
    with open(class_names_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_inference(image_path: Path, checkpoint_path: Path, class_names: list) -> dict:
    # 전체 로직을 try-except로 감싸 추론 중 발생하는 모든 에러 방어
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = build_efficientnet_b3(num_classes=len(class_names), device=device, pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        transform = transforms.Compose([
            ReinhardNormalizer(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            
        pred_class = int(torch.argmax(prob).item())
        confidence = float(prob[pred_class].item())
        
        return {
            "pred_class": pred_class,
            "pred_label": class_names[pred_class],
            "probabilities": [float(p) for p in prob.tolist()],
            "confidence": confidence
        }
        
    except Exception as e:
        # 에러 발생 시 예외 정보를 딕셔너리로 반환
        return {
            "error": "Inference failed",
            "message": str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="피부 병변 단일 이미지 추론")
    parser.add_argument("--image", type=Path, required=True, help="추론할 이미지 경로 (예: ./sample.jpg)")
    parser.add_argument("--checkpoint", type=Path, default=Path("./outputs/best_model.pth"), help="모델 체크포인트 경로")
    parser.add_argument("--class-names", type=Path, default=Path("./outputs/class_names.json"), help="클래스 이름 JSON 파일 경로")
    args = parser.parse_args()
    
    if not args.image.exists():
        print(json.dumps({"error": f"이미지를 찾을 수 없습니다 - {args.image}"}, ensure_ascii=False))
        exit(1)
    if not args.checkpoint.exists():
        print(json.dumps({"error": f"모델 가중치를 찾을 수 없습니다 - {args.checkpoint}"}, ensure_ascii=False))
        exit(1)
    if not args.class_names.exists():
        print(json.dumps({"error": f"클래스 이름 파일을 찾을 수 없습니다 - {args.class_names}"}, ensure_ascii=False))
        exit(1)

    # 실행 및 안전한 JSON 출력
    class_names = load_class_names(args.class_names)
    result = run_inference(args.image, args.checkpoint, class_names)
    print(json.dumps(result, indent=4, ensure_ascii=False))