#!/bin/bash
#set -e
cd "$(dirname "$0")"

# 기존 venv 재사용
#source .venv/bin/activate

python run_ani.py --camera 0 --width 640 --height 480 --win_w 1280 --win_h 720 --topk 3 --detail_topk 10

