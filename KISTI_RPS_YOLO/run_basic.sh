#!/bin/bash
# 기본 가위바위보 게임 모드
# 사용 전: source .venv/bin/activate 실행되어 있어야 함

cd "$(dirname "$0")"   # 현재 스크립트 위치로 이동

# USB 카메라는 /dev/video0 기준
python run_game.py --camera 0