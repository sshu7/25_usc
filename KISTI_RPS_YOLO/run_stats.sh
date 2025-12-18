#!/bin/bash
# 전적(승/무/패) 카운트가 화면 상단 HUD에 표시되는 모드
# 파이썬 파일: run_game_stats.py (기존 run_game.py 확장 버전)

cd "$(dirname "$0")"

python run_game_stats.py --camera 0
