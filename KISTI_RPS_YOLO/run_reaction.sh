#!/bin/bash
# 반응속도(Reaction Time) 모드
# 화면에 "ROCK / PAPER / SCISSORS" 중 하나가 랜덤으로 나오고,
# 학생이 해당 손모양을 맞췄을 때 걸린 시간을 재서 화면에 보여주는 버전.

cd "$(dirname "$0")"

python run_game_reaction.py --camera 0
