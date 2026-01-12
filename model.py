import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from typing import Tuple, List, Dict

# dataloader.py와 visualize.py에서 정의한 상수 및 함수가 필요합니다.
from dataloader import DataManager, create_dataset, GENRE_LIST, TARGET_SIZE

# --- [1] 다중 레이블 커스텀 지표 ---

def exact_match_ratio(y_true, y_pred):
    """모든 장르를 완벽하게 맞춘 비율 (Strict Accuracy)"""
    threshold = 0.5
    y_pred_bin = K.cast(K.greater(y_pred, threshold), 'float32')
    match = K.all(K.equal(y_true, y_pred_bin), axis=1)
    return K.mean(match)

def micro_f1_score(y_true, y_pred):
    """전체 레이블 합산 기준 F1-Score (Micro평균)"""
    threshold = 0.5
    y_pred_bin = K.cast(K.greater(y_pred, threshold), 'float32')
    
    tp = K.sum(y_true * y_pred_bin)
    fp = K.sum((1 - y_true) * y_pred_bin)
    fn = K.sum(y_true * (1 - y_pred_bin))
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def jaccard_similarity(y_true, y_pred):
    """실제 레이블과 예측 레이블 간의 교집합/합집합 비율 (Intersection over Union)"""
    threshold = 0.5
    y_pred_bin = K.cast(K.greater(y_pred, threshold), 'float32')
    
    intersection = K.sum(y_true * y_pred_bin, axis=1)
    union = K.sum(K.clip(y_true + y_pred_bin, 0, 1), axis=1)
    return K.mean(intersection / (union + K.epsilon()))

# --- [2] 모델 구조 정의 (VGG-Style CNN) ---



def build_cnn_model(input_shape: Tuple[int, int, int], num_genres: int):
    """다중 레이블 분류를 위한 CNN 모델 생성 및 컴파일"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        # 다중 레이블을 위해 sigmoid 사용
        layers.Dense(num_genres, activation='sigmoid') 
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # Multi-label 핵심 설정
        metrics=[
            'accuracy',
            exact_match_ratio,
            micro_f1_score,
            jaccard_similarity,
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# --- [3] 학습 실행 함수 ---

def run_training(epochs: int, batch_size: int, samples_per_genre: int, base_dir: str):
    """학습 데이터 샘플링 및 모델 훈련 루프 실행"""
    
    # 1. 데이터 매니저 초기화 (훈련용)
    train_manager = DataManager(
        csv_path=os.path.join(base_dir, 'training_tag_vectors.csv'),
        data_root_dir=os.path.join(base_dir, 'Processed_224x224')
    )
    
    # 2. 검증 데이터셋 준비 (검증은 한 번만 로드하여 고정 사용)
    val_manager = DataManager(
        csv_path=os.path.join(base_dir, 'validation_tag_vectors.csv'),
        data_root_dir=os.path.join(base_dir, 'Validation_Set')
    )
    v_paths, v_tags = val_manager.get_dataset_lists() # 모든 데이터 로드
    val_dataset = create_dataset(v_paths, v_tags, batch_size, is_training=False)
    
    # 3. 모델 구축
    model = build_cnn_model(input_shape=(*TARGET_SIZE, 3), num_genres=len(GENRE_LIST))
    
    # 학습 결과 기록용
    final_history = {
        'loss': [], 'val_loss': [], 
        'accuracy': [], 'val_accuracy': [],
        'micro_f1_score': [], 'val_micro_f1_score': []
    }

    print(f"\n🚀 학습 시작 (총 {epochs} Epochs, 에폭당 장르별 {samples_per_genre}개 샘플링)")
    
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        
        # [핵심] 매 에폭마다 장르별로 동일한 수만큼 무작위 샘플링 (데이터 균형화)
        t_paths, t_tags = train_manager.get_dataset_lists(samples_per_genre=samples_per_genre)
        train_dataset = create_dataset(t_paths, t_tags, batch_size, is_training=True)
        
        # 훈련 수행 (1에폭씩)
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1,
            verbose=1
        )
        
        # 기록 업데이트
        for key in final_history.keys():
            if key in history.history:
                final_history[key].append(history.history[key][0])

    print("\n✅ 모든 학습 과정이 완료되었습니다.")
    return model, final_history, val_dataset