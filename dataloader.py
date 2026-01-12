import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from typing import List, Tuple, Dict

# --- [1] 설정값 및 상수 ---
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
GENRE_LIST: List[str] = [
    "Adventure", "Action", "RPG", "Strategy", "Simulation", "Sports", "Racing", 
    "Puzzle", "Sandbox", "Shooter", "Survival"
]

# --- [2] 데이터 관리 클래스 ---

class DataManager:
    """파일 경로 캐싱 및 장르별 샘플링을 담당합니다."""
    def __init__(self, csv_path: str, data_root_dir: str):
        self.data_root_dir = data_root_dir
        tag_df = pd.read_csv(csv_path).set_index('filename')
        
        self.filename_to_path = {}
        self.tag_vectors_map = {}
        self.genre_to_filenames = {genre: [] for genre in GENRE_LIST}

        print(f"🚚 데이터 매핑 로딩 중: {os.path.basename(csv_path)}")
        
        for filename, row in tag_df.iterrows():
            tag_vector = row[GENRE_LIST].values.astype(np.float32)
            
            # 파일이 존재하는 장르 폴더 탐색 (최적화: 첫 번째 일치 장르 사용)
            path_found = None
            active_genres = [g for g in GENRE_LIST if row[g] == 1]
            
            for genre in active_genres:
                tmp_path = os.path.join(data_root_dir, genre, filename)
                if os.path.exists(tmp_path):
                    path_found = tmp_path
                    break
            
            if path_found:
                self.filename_to_path[filename] = path_found
                self.tag_vectors_map[filename] = tag_vector
                for genre in active_genres:
                    self.genre_to_filenames[genre].append(filename)

    def get_dataset_lists(self, samples_per_genre: int = None) -> Tuple[List[str], List[np.ndarray]]:
        """
        데이터셋 생성을 위한 경로와 태그 리스트를 반환합니다.
        samples_per_genre가 None이면 모든 데이터를 반환합니다 (검증용).
        """
        paths, tags = [], []

        for genre in GENRE_LIST:
            fnames = self.genre_to_filenames[genre]
            if not fnames: continue

            if samples_per_genre: # 훈련 시: 균형 샘플링 (복원 추출 포함)
                selected = random.choices(fnames, k=samples_per_genre)
            else: # 검증 시: 해당 장르의 모든 파일 (중복 제거 필요 시 추가 로직 가능)
                selected = fnames
                
            for f in selected:
                paths.append(self.filename_to_path[f])
                tags.append(self.tag_vectors_map[f])
        
        return paths, tags

# --- [3] tf.data 파이프라인 함수 ---

@tf.function
def apply_augmentation(img):
    """안정적인 tf.image 함수를 이용한 데이터 증강."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img

def load_and_preprocess_image(file_path, tag_vector, is_training):
    """이미지 로드, 전처리 및 조건부 증강."""
    img_raw = tf.io.read_file(file_path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, TARGET_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    # 훈련 모드일 때만 증강 적용
    if is_training:
        img = apply_augmentation(img)
        
    return img, tag_vector

def create_dataset(paths_list, tags_list, batch_size: int, is_training: bool):
    """tf.data.Dataset 객체 생성 및 최적화."""
    dataset = tf.data.Dataset.from_tensor_slices((paths_list, tags_list))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths_list), 5000))
        
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- [4] 사용 예시 (Main) ---
if __name__ == "__main__":
    BASE_DIR = r'F:\ML\dataset'
    TRAIN_CSV = os.path.join(BASE_DIR, 'final_unique_tag_vectors.csv')
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'Processed_224x224')

    # 매니저 초기화 (한 번만 수행)
    manager = DataManager(TRAIN_CSV, TRAIN_IMG_DIR)

    # 에폭마다 호출할 리스트 생성 (장르당 1000개씩 샘플링 예시)
    p_list, t_list = manager.get_dataset_lists(samples_per_genre=1000)
    
    # 데이터셋 생성
    train_ds = create_dataset(p_list, t_list, batch_size=BATCH_SIZE, is_training=True)
    print("✅ 데이터셋 생성 완료")