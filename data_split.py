import os
import random
import shutil
import glob
import pandas as pd
from typing import List, Dict, Set

# --- [1] 통합 설정값 ---
BASE_DIR = r'F:\ML\dataset'
SOURCE_DIR_NAME = 'Processed_224x224'    # 원본(학습용) 이미지 폴더
VALIDATION_DIR_NAME = 'Validation_Set'   # 검증용 이미지 저장 폴더
FILES_TO_MOVE_PER_GENRE = 3000           # 장르별 추출할 이미지 수

FINAL_CSV_FILENAME = 'final_unique_tag_vectors.csv'     # 전체 통합 CSV
VALIDATION_CSV_FILENAME = 'validation_tag_vectors.csv'  # 검증셋 전용 CSV

TARGET_TAGS = [
    "Adventure", "Action", "RPG", "Strategy", "Simulation", "Sports", "Racing", 
    "Puzzle", "Sandbox", "Shooter", "Survival"
]

# --- [2] 헬퍼 함수 (유틸리티) ---

def get_file_count_and_sort_tags(source_base_dir: str) -> List[str]:
    """장르별 파일 수를 확인하고 데이터가 적은 순으로 정렬합니다."""
    tag_counts = {}
    for tag in TARGET_TAGS:
        path = os.path.join(source_base_dir, tag)
        if os.path.isdir(path):
            tag_counts[tag] = len(glob.glob(os.path.join(path, '*.[pj][np]g')))
        else:
            tag_counts[tag] = 0
    
    sorted_tags = sorted(tag_counts.keys(), key=lambda t: tag_counts[t])
    print("📊 장르별 데이터 현황 (오름차순):")
    for tag in sorted_tags:
        print(f"  - {tag}: {tag_counts[tag]}개")
    return sorted_tags

def remove_duplicates_from_all_train_dirs(filename: str):
    """검증셋으로 뽑힌 파일이 다른 학습용 장르 폴더에 남아있지 않도록 제거합니다."""
    source_base_dir = os.path.join(BASE_DIR, SOURCE_DIR_NAME)
    for tag in TARGET_TAGS:
        file_path = os.path.join(source_base_dir, tag, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  [오류] 중복 제거 실패 ({filename}): {e}")

# --- [3] 핵심 기능 1: 이미지 분리 (Validation Set 구축) ---

def run_data_split():
    """학습 폴더에서 이미지를 랜덤 추출하여 검증 폴더로 이동시킵니다."""
    source_base_dir = os.path.join(BASE_DIR, SOURCE_DIR_NAME)
    validation_base_dir = os.path.join(BASE_DIR, VALIDATION_DIR_NAME)
    os.makedirs(validation_base_dir, exist_ok=True)

    sorted_tags = get_file_count_and_sort_tags(source_base_dir)
    global_validation_filenames: Set[str] = set()

    # 기존 검증 폴더에 파일이 있다면 중복 방지를 위해 추적 셋에 추가
    for tag in TARGET_TAGS:
        v_path = os.path.join(validation_base_dir, tag)
        if os.path.isdir(v_path):
            files = glob.glob(os.path.join(v_path, '*.[pj][np]g'))
            global_validation_filenames.update({os.path.basename(f) for f in files})

    print("-" * 50)
    print(f"🚀 검증셋 분리 시작 (목표: 장르별 {FILES_TO_MOVE_PER_GENRE}개)")
    
    for tag in sorted_tags:
        s_dir = os.path.join(source_base_dir, tag)
        v_dir = os.path.join(validation_base_dir, tag)
        os.makedirs(v_dir, exist_ok=True)

        current_v_count = len(glob.glob(os.path.join(v_dir, '*.[pj][np]g')))
        if current_v_count >= FILES_TO_MOVE_PER_GENRE:
            print(f"[{tag}] ✅ 이미 목표치 달성. 스킵.")
            continue

        needed = FILES_TO_MOVE_PER_GENRE - current_v_count
        all_files = glob.glob(os.path.join(s_dir, '*.[pj][np]g'))
        
        # 다른 장르에서 이미 뽑힌 파일 제외
        candidates = [f for f in all_files if os.path.basename(f) not in global_validation_filenames]

        if not candidates:
            print(f"[{tag}] ⚠️ 이동할 수 있는 고유 파일이 없습니다.")
            continue

        selected = random.sample(candidates, min(len(candidates), needed))
        
        moved_count = 0
        for s_path in selected:
            fname = os.path.basename(s_path)
            d_path = os.path.join(v_dir, fname)
            try:
                # 1. 파일 이동
                shutil.move(s_path, d_path)
                # 2. 다른 학습 폴더 내 동일 파일 삭제 (데이터 오염 방지)
                remove_duplicates_from_all_train_dirs(fname)
                global_validation_filenames.add(fname)
                moved_count += 1
            except Exception as e:
                print(f"  [오류] {fname} 이동 실패: {e}")

        print(f"[{tag}] 완료: {current_v_count + moved_count}개 확보 (이번에 {moved_count}개 이동)")

    print(f"✅ 검증셋 이미지 분리 완료. 총 고유 이미지: {len(global_validation_filenames)}장")

# --- [4] 핵심 기능 2: 검증셋용 CSV 생성 ---

def run_create_validation_csv():
    """실제 검증 폴더에 있는 파일들만 필터링하여 전용 CSV를 생성합니다."""
    print("-" * 50)
    print("📊 검증셋 전용 CSV 생성 시작...")
    
    v_base_dir = os.path.join(BASE_DIR, VALIDATION_DIR_NAME)
    source_csv_path = os.path.join(BASE_DIR, FINAL_CSV_FILENAME)
    output_csv_path = os.path.join(BASE_DIR, VALIDATION_CSV_FILENAME)

    # 1. 실제 폴더 내 파일명 수집
    search_pattern = os.path.join(v_base_dir, '**', '*.[pj][np]g')
    v_files = glob.glob(search_pattern, recursive=True)
    v_filenames = {os.path.basename(f) for f in v_files}

    if not v_filenames:
        print("❌ 오류: 검증 폴더에 이미지가 없습니다."); return

    # 2. 원본 CSV 로드 및 필터링
    try:
        df_full = pd.read_csv(source_csv_path)
        df_val = df_full[df_full['filename'].isin(v_filenames)]
        
        df_val.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"✅ 검증용 CSV 저장 완료: {output_csv_path}")
        print(f"   (매칭된 행 수: {len(df_val)} / 실제 파일 수: {len(v_filenames)})")
    except Exception as e:
        print(f"❌ CSV 생성 실패: {e}")

# --- [5] 실행 제어 ---

if __name__ == "__main__":
    # 1. 학습/검증 데이터 물리적 분리 실행
    run_data_split()
    
    # 2. 분리된 데이터를 바탕으로 검증용 CSV 생성
    run_create_validation_csv()