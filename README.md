# 26.1_skin_analyzing
AI-based Skin Lesion Classification System (13 Classes) using EfficientNet-B3 &amp; PyTorch

1. 목표
로컬 데이터셋과 외부(Kaggle) 데이터셋을 통합하여 13종의 피부 병변 및 상태를 정확하게 분류하는 딥러닝 모델 개발
다양한 촬영 환경에서 발생하는 색감 차이를 극복하기 위해 통계적 색상 정규화(Reinhard Normalization)를 적용한 강건한 파이프라인 구축

2. 데이터셋
로컬 데이터셋: 01.원천데이터(이미지)와 02.라벨링데이터(JSON)로 구성되며, 전문적인 진단 정보(diagnosis_name)를 포함함.
Kaggle 데이터셋: nayanchaure/acne-dataset을 활용하며, 여드름의 중증도(Level 0~3) 데이터를 포함함

4. 클래스 목록 (Total: 13)로컬 데이터셋의 19개 범주를 인덱싱 최적화를 통해 13개 클래스로 매핑함
주요 포함 항목: Whitehead, Blackhead, Papule, Pustule, Nodule, Sebaceous Calculi, Milia, Syringoma, Enlarged Pores, Melasma, PIH, Rosacea, Seborrheic Dermatitis 등

5. 학습 파이프라인
   초기화: GPU/CPU 디바이스 설정 및 하이퍼파라미터 로드
   데이터 로딩: 로컬과 Kaggle 데이터를 각각의 Dataset 객체로 생성 후 ConcatDataset으로 병합
   실시간 전처리: 데이터를 불러오는 시점에 Reinhard 정규화 및 텐서 변환 적용
   모델 학습: EfficientNet-B3 백본을 사용하여 Focal Loss 기반의 최적화 진행저장: 학습된 가중치를 .pth 파일로 출력

6. 기본 전처리 및 증강
  Reinhard 정규화: L*a*b 색 공간에서 평균과 표준편차를 타겟 통계치($L: 150, a/b: 128$)에 맞춰 조정
  크기 조정: 모든 이미지를 $300 \times 300$ 픽셀로 리사이징
  정규화: ImageNet 데이터셋의 평균과 표준편차를 적용

7. 모델 구성
   백본(Backbone): EfficientNet-B3
   가중치: IMAGENET1K_V1 사전 학습 가중치 사용
   헤드(Head):Dropout (p=0.3) 적용, 최종 출력층을 13개 클래스에 맞게 수정

8. 평가지표손실 함수(Loss Function):
   Focal Loss (Alpha=1.0, Gamma=2.0), 클래스 불균형이 심한 의료 데이터 특성을 고려하여 어려운 샘플에 가중치를 부여함
   학습 모니터링: 에폭(Epoch) 및 스텝(Step)별 실시간 손실값(Loss Item)

9. 결과 파일skin_final_gpu.pth: 학습이 완료된 모델의 최종 가중치 파일
