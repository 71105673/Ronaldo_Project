# score_manager.py

import os

def load_highscore():
    """최고 기록을 파일에서 불러옵니다."""
    if not os.path.exists("highscore.txt"): return 0
    try:
        with open("highscore.txt", "r") as f: return int(f.read())
    except (IOError, ValueError): return 0

def save_highscore(new_score):
    """새로운 최고 기록을 파일에 저장합니다."""
    try:
        with open("highscore.txt", "w") as f: f.write(str(new_score))
    except IOError as e: print(f"최고 기록 저장 오류: {e}")