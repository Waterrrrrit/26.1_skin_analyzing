# Lesion Classification (Skin Lesion 13-Class)

이 프로젝트는 일반 피부 병변 이미지 데이터를 이용해 13개 병변을 분류하는 baseline 학습 파이프라인입니다. "데이터 읽기 → 전처리/증강 → 모델 학습 → 검증 → 결과 저장"의 전체 흐름을 한 번에 실행할 수 있도록 구성되어 있습니다.

## 목표

* 일반 피부 병변 13개 클래스 분류 baseline 구축
* 전이학습 기반 EfficientNet-B3 모델을 사용하여 미세한 피부 병변에 대한 고해상도 특징 추출 성능 확보
* Validation 분리 기반 정량 평가(accuracy/precision/recall/f1) 수행 및 Focal Loss를 통한 데이터 불균형 문제 완화

## 데이터셋 설명

* **데이터 루트**: `dataset/`
* **split**: `Training`, `Validation`
* **이미지 폴더**: `images`
* **라벨 폴더**: `jsons`
* 이미지 파일과 JSON 라벨 파일은 동일한 stem을 공유
* JSON 내부 `diagnosis_name`을 정답 라벨로 사용
* **클래스 수**: 13
* **출처**: AI Hub, Kaggle 통합 데이터

## 클래스 목록

* 화이트헤드 (Whitehead)
* 블랙헤드 (Blackhead)
* 구진 (Papule)
* 농포 (Pustule)
* 결절 (Nodule)
* 피지선결석 (Sebaceous Calculus)
* 비립종 (Milium)
* 한관종 (Syringoma)
* 모공확장 (Enlarged Pores)
* 기미 (Melasma)
* 색소침착 (Pigmentation)
* 주사(딸기코) (Rosacea)
* 지루성 피부염 (Seborrheic Dermatitis)

## 클래스별 샘플 수

* **Train**: 각 클래스 000장 *(※ 본인이 구축한 실제 데이터셋 장수에 맞게 숫자를 수정해 주세요)*
* **Val**: 각 클래스 00장 *(※ 본인이 구축한 실제 데이터셋 장수에 맞게 숫자를 수정해 주세요)*
