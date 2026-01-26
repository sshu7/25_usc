#!/bin/bash
# 실시간 동물 유사도 분석 모드

cd "$(dirname "$0")"

# 기존 가위바위보와 동일한 venv 사용
#source .venv/bin/activate

python run_everything.py --camera 0 --width 640 --height 480
