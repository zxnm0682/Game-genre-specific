import json
import requests
import os
import glob
import time
import random
import concurrent.futures
import pandas as pd
from urllib.parse import urlparse
from typing import Set, List, Dict, Tuple, Any
from PIL import Image, ImageFile

# --- [1] 통합 설정값 (한 곳에서 관리) ---
FILE_PATH = 'games.json'              # 원본 데이터 경로
BASE_DIR = r'F:\ML\dataset'          # 기본 작업 경로
RESIZED_DIR_NAME = 'Processed_224x224'
FINAL_CSV_FILENAME = 'final_unique_tag_vectors.csv'
CSV_PREFIX = 'screenshot_tag_vectors_'

TARGET_GENRE = "Survival"             # 수집할 메인 장르
TARGET_MAX_FILES = 32000              # 장르별 최대 파일 수
TARGET_SIZE = (224, 224)              # 리사이징 크기

TARGET_TAGS = [
    "Adventure", "Action", "RPG", "Strategy", "Simulation", "Sports", "Racing", 
    "Puzzle", "Sandbox", "Shooter", "Survival", "Fighting", "Music"
]

MAX_WORKERS_DL = 30    # 다운로드용 스레드 수
MAX_PROCESSES_RESIZE = 4 # 리사이징용 프로세스 수

# 이미지 손상 방지 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- [2] 헬퍼 함수 (유틸리티) ---

def clean_filename(url: str) -> str:
    path = urlparse(url).path
    filename = os.path.basename(path)
    if '?' in filename:
        filename = filename.split('?')[0]
    return filename

def get_genre_file_count(genre_dir: str) -> int:
    if not os.path.exists(genre_dir): return 0
    return len([n for n in os.listdir(genre_dir) if os.path.isfile(os.path.join(genre_dir, n)) and not n.startswith('.')])

def manage_file_limit(genre_dir: str, max_limit: int):
    current_count = get_genre_file_count(genre_dir)
    if current_count > max_limit:
        files_to_delete = current_count - max_limit
        all_files = [f for f in os.listdir(genre_dir) if os.path.isfile(os.path.join(genre_dir, f)) and not f.startswith('.')]
        files_to_remove = random.sample(all_files, files_to_delete)
        print(f"[{os.path.basename(genre_dir)}] ❗️ 한도 초과: {files_to_delete}개 삭제 중...")
        for file in files_to_remove:
            try: os.remove(os.path.join(genre_dir, file))
            except OSError: pass

def create_tag_vector(matching_tags: Set[str]) -> List[int]:
    return [1 if tag in matching_tags else 0 for tag in TARGET_TAGS]

# --- [3] 핵심 기능 1: 데이터 다운로드 ---

def download_screenshot(url: str, genre_folder: str, appid: str) -> Tuple[bool, str]:
    filename = f"{appid}_{clean_filename(url)}"
    save_path = os.path.join(genre_folder, filename)
    if os.path.exists(save_path): return True, filename
    try:
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)
        return True, filename
    except: return False, filename

def run_crawler():
    """JSON을 읽어 스크린샷을 다운로드하고 태그 CSV를 생성합니다."""
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f: data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 로드 실패: {e}"); return

    genre_dir = os.path.join(BASE_DIR, TARGET_GENRE)
    os.makedirs(genre_dir, exist_ok=True)
    manage_file_limit(genre_dir, TARGET_MAX_FILES)
    
    current_count = get_genre_file_count(genre_dir)
    csv_data_map, total_downloads = {}, 0
    
    print(f"🚀 크롤링 시작: {TARGET_GENRE} (목표: {TARGET_MAX_FILES})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_DL) as executor:
        future_to_filename = {}
        for appid, game in data.items():
            if current_count >= TARGET_MAX_FILES: break
            game_tags = set(game.get('tags', {}).keys() if isinstance(game.get('tags'), dict) else game.get('tags', []))
            
            if TARGET_GENRE not in game_tags: continue
            
            tag_vector = create_tag_vector(game_tags.intersection(TARGET_TAGS))
            for shot in game.get('screenshots', []):
                url = shot.get('path_full') if isinstance(shot, dict) else shot
                if url:
                    fname = f"{appid}_{clean_filename(url)}"
                    csv_data_map[fname] = tag_vector
                    future = executor.submit(download_screenshot, url, genre_dir, appid)
                    future_to_filename[future] = fname

        downloaded_files = set()
        for future in concurrent.futures.as_completed(future_to_filename):
            is_success, filename = future.result()
            if is_success:
                total_downloads += 1
                downloaded_files.add(filename)
                if total_downloads % 100 == 0: print(f"--- 현재 {total_downloads}개 완료 ---")

    # CSV 저장
    records = [{'filename': f, 'tag_vector': v} for f, v in csv_data_map.items() if f in downloaded_files]
    if records:
        df = pd.DataFrame(records)
        df_final = pd.concat([df[['filename']], pd.DataFrame(df['tag_vector'].tolist(), columns=TARGET_TAGS)], axis=1)
        df_final.to_csv(os.path.join(BASE_DIR, f'{CSV_PREFIX}{TARGET_GENRE}.csv'), index=False)
    print(f"✅ 크롤링 종료. 총 {total_downloads}개 저장 완료.")

# --- [4] 핵심 기능 2: 이미지 전처리 ---

def pad_to_square(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    width, height = img.size
    ratio = min(target_size[0] / width, target_size[1] / height)
    new_w, new_h = int(width * ratio), int(height * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    new_img.paste(img, ((target_size[0]-new_w)//2, (target_size[1]-new_h)//2))
    return new_img

def process_single_image(args):
    in_p, out_p, size = args
    try:
        if os.path.exists(out_p): return True
        img = Image.open(in_p).convert('RGB')
        pad_to_square(img, size).save(out_p, format='JPEG', quality=90)
        return True
    except: return False

def run_preprocessing():
    output_base = os.path.join(BASE_DIR, RESIZED_DIR_NAME)
    all_tasks = []
    for tag in TARGET_TAGS:
        in_dir = os.path.join(BASE_DIR, tag)
        out_dir = os.path.join(output_base, tag)
        if not os.path.isdir(in_dir): continue
        os.makedirs(out_dir, exist_ok=True)
        for f in glob.glob(os.path.join(in_dir, '*.[pj][np]g')):
            all_tasks.append((f, os.path.join(out_dir, os.path.basename(f)), TARGET_SIZE))
    
    print(f"🖼 전처리 시작 (대상: {len(all_tasks)}개)")
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSES_RESIZE) as executor:
        results = list(executor.map(process_single_image, all_tasks))
    print(f"✅ 전처리 완료: {sum(results)}개 성공.")

# --- [5] 핵심 기능 3: CSV 통합 ---

def merge_csv_files():
    search_p = os.path.join(BASE_DIR, f'{CSV_PREFIX}*.csv')
    csv_files = glob.glob(search_p)
    if not csv_files: return
    
    print(f"🔍 CSV 통합 중... ({len(csv_files)}개)")
    df_list = [pd.read_csv(f) for f in csv_files]
    df_combined = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['filename'])
    df_combined.to_csv(os.path.join(BASE_DIR, FINAL_CSV_FILENAME), index=False)
    print(f"✅ 통합 완료. 총 {len(df_combined)}행 저장.")

# --- [실행 제어] ---
if __name__ == "__main__":
    # 순서대로 실행하려면 주석을 해제하세요.
    # run_crawler()         # 1. 크롤링
    # run_preprocessing()   # 2. 리사이징
    # merge_csv_files()     # 3. CSV 통합
    pass