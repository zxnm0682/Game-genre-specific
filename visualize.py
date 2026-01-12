import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.callbacks import History
from sklearn.metrics import confusion_matrix
from typing import List, Union

# --- [1] 기본 설정 (필요 시 수정) ---
# dataloader.py의 GENRE_LIST와 동일해야 합니다.
DEFAULT_GENRES = [
    "Adventure", "Action", "RPG", "Strategy", "Simulation", "Sports", "Racing", 
    "Puzzle", "Sandbox", "Shooter", "Survival"
]

# --- [2] 학습 곡선 시각화 ---


def plot_loss_curve(history: keras.callbacks.History):
    """Keras History 객체에서 손실(loss) 곡선을 시각화합니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_curve(history: keras.callbacks.History):
    """Keras History 객체에서 정확도(accuracy) 곡선을 시각화합니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- [3] 다중 레이블 컨퓨전 매트릭스 ---

def plot_genre_confusion_matrices(y_true: np.ndarray, 
                                 y_pred_prob: np.ndarray, 
                                 genre_list: List[str] = DEFAULT_GENRES, 
                                 threshold: float = 0.5):
    """
    다중 레이블 결과를 기반으로 각 장르별 2x2 이진 Confusion Matrix를 시각화합니다.
    """
    # 확률값을 임계값 기준으로 0 또는 1로 변환
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    n_genres = len(genre_list)
    
    # 출력 그리드 설정 (4열 구성)
    cols = 4
    rows = (n_genres + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    print("\n" + "="*50)
    print(f"📈 장르별 성능 상세 지표 (Threshold: {threshold})")
    print("="*50)

    for i, genre in enumerate(genre_list):
        # 특정 장르 열 추출
        gt = y_true[:, i]
        pred = y_pred_bin[:, i]
        
        # CM 계산: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(gt, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # 콘솔 출력 (디버깅용)
        print(f"[{genre:12}] | TP: {tp:5} | TN: {tn:5} | FP: {fp:5} | FN: {fn:5}")

        # 히트맵 시각화
        df_cm = pd.DataFrame(cm, index=['False', 'True'], columns=['Pred False', 'Pred True'])
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'Genre: {genre}', fontsize=12, fontweight='bold')
        
    # 남는 서브플롯 제거
    for j in range(n_genres, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    print("="*50 + "\n")

# --- [4] 통합 검증 함수 ---

def evaluate_model_visual(model, dataset, genre_list: List[str] = DEFAULT_GENRES):
    """
    모델과 tf.data.Dataset을 받아 예측을 수행하고 결과를 시각화합니다.
    """
    print("⏳ 모델 예측 수행 중 (검증 데이터셋)...")
    
    # 데이터셋에서 실제값(y_true)과 예측값(y_pred) 추출
    all_y_true = []
    for _, y in dataset:
        all_y_true.append(y.numpy())
    y_true = np.concatenate(all_y_true, axis=0)
    
    # 예측 수행
    y_pred_prob = model.predict(dataset, verbose=1)
    
    # CM 플롯 출력
    plot_genre_confusion_matrices(y_true, y_pred_prob, genre_list=genre_list)