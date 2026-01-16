# 게임 장르 다중 레이블 분류 CNN (Game Genre Multi-label Classification)

이 프로젝트는 게임 스크린샷 이미지를 분석하여 해당 게임이 속한 여러 장르(Action, RPG, Strategy 등)를 동시에 예측하는 Deep Learning 기반 다중 레이블 분류 시스템입니다.

**1. data_pipeline.py & data_split.py**
   
전처리: 고해상도 이미지를 모델 입력 크기(224x224)로 리사이징하고 픽셀 값을 정규화합니다.

데이터셋 분리: 학습 데이터와 검증 데이터 간의 폴더 구조를 분리해 구성합니다

CSV 동기화: 이미지 파일명과 다중 레이블(One-hot encoding) 벡터를 매핑한 통합 CSV를 생성합니다.

**2. dataloader.py**
   
동적 샘플링: 특정 장르에 데이터가 치우치지 않도록 매 에폭마다 각 장르에서 동일한 수의 샘플을 무작위로 추출하여 학습에 사용합니다.

**3. model.py (Model Architecture)**
   
CNN 기반 설계: VGG-Style의 합성곱 신경망을 기반으로 Batch Normalization과 Dropout을 적용하여 과적합을 방지했습니다.

다중 레이블 분류: 출력층에 Sigmoid 활성화 함수를, 손실 함수로 Binary Crossentropy를 사용하여 각 장르의 독립적인 존재 확률을 예측합니다.

커스텀 지표: 단순 정확도 대신 Exact Match Ratio, Micro F1-Score를 구현하여 다중 레이블 모델의 성능을 정밀하게 측정합니다.

**4. visualize.py**
   
학습 모니터링: Loss, Accuracy의 변화 추이를 시각화합니다.

장르별 성능 분석: 전체 평균이 아닌 개별 장르별 2x2 Confusion Matrix를 히트맵으로 출력하여, 모델이 특히 약한 장르를 파악하고 개선 방향을 제시합니다.

**5. main.py & Notebooks**
   
파이프라인 제어: 데이터 분리부터 학습, 모델 저장까지의 전 과정을 자동화했습니다.

실험 기록: notebooks/ 폴더 내의 파일들을 통해 초기 아이디어 구상부터 최종 모델 완성까지의 실험 결과와 시각화 로그를 확인할 수 있습니다.

📊 Dataset Information

이 프로젝트에 사용된 데이터셋은 https://www.kaggle.com/datasets/fronkongames/steam-games-datasetd 의 'games.json' 파일을 이용해 실험 목적에 맞게 서브셋을 추출하여 사용했습니다.
