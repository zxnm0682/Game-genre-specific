import os
import tensorflow as tf
from data_split import run_data_split, run_create_validation_csv
from model import run_training
from visualize import plot_history, evaluate_model_visual

# --- [1] 전체 실행 설정 ---
CONFIG = {
    "BASE_DIR": r'F:\ML\dataset',
    "EPOCHS": 20,
    "BATCH_SIZE": 32,
    "SAMPLES_PER_EPOCH_PER_GENRE": 10000,  # 에폭당 장르별 추출 이미지 수
    "MODEL_SAVE_NAME": "game_genre_cnn_v1.h5",
    "DO_DATA_SPLIT": False  # 이미 데이터를 나누었다면 False로 변경하세요.
}

def main():
    # --- [Step 1] 데이터셋 준비 (최초 1회 실행 필요) ---
    if CONFIG["DO_DATA_SPLIT"]:
        print("\n=== [STEP 1] 데이터셋 물리적 분리 및 CSV 생성 ===")
        run_data_split()           # 이미지 이동 및 중복 제거
        run_create_validation_csv() # 검증용 CSV 생성
    else:
        print("\n=== [STEP 1] 데이터셋 준비 스킵 (기존 데이터 사용) ===")

    # --- [Step 2] 모델 학습 시작 ---
    print("\n=== [STEP 2] 모델 학습 및 검증 루프 시작 ===")
    # run_training은 model, history, val_dataset을 반환합니다.
    model, history, val_dataset = run_training(
        epochs=CONFIG["EPOCHS"],
        batch_size=CONFIG["BATCH_SIZE"],
        samples_per_genre=CONFIG["SAMPLES_PER_EPOCH_PER_GENRE"],
        base_dir=CONFIG["BASE_DIR"]
    )

    # --- [Step 3] 학습 결과 시각화 ---
    print("\n=== [STEP 3] 학습 곡선 및 성능 시각화 ===")
    # 손실, 정확도 곡선 출력
    plot_loss_curve(model_history)
    plot_accuracy_curve(model_history)
    
    # 상세 컨퓨전 매트릭스 (장르별 성능) 출력
    evaluate_model_visual(model, val_dataset)

    # --- [Step 4] 모델 저장 ---
    print("\n=== [STEP 4] 모델 저장 ===")
    save_path = os.path.join(CONFIG["BASE_DIR"], CONFIG["MODEL_SAVE_NAME"])
    model.save(save_path)
    print(f"✅ 모델이 성공적으로 저장되었습니다: {save_path}")

if __name__ == "__main__":
    # GPU 메모리 효율화 설정 (필요 시)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main()