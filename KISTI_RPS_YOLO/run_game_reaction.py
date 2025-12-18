# run_game_reaction.py
import cv2
import time
import random
import argparse

from utils.pipeline import RPSPipeline
from utils.visualization import draw_bbox_label, draw_reaction_info
from utils.camera import Camera

GESTURES = ["rock", "paper", "scissors"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    cam = Camera(args.camera)
    pipe = RPSPipeline()

    target = None           # 현재 맞춰야 할 제스처
    start_time = None       # 타이머 시작 시각
    reaction_time = None    # 마지막 반응속도 기록(초)

    print("=== RPS Reaction Mode ===")
    print("  - R: 새로운 목표 제스처 생성")
    print("  - Q: 종료")
    print("==========================")

    while True:
        frame = cam.read()
        if frame is None:
            print("❗ 카메라 프레임을 읽지 못했습니다. 종료합니다.")
            break

        # YOLO + 제스처 판별
        frame, info = pipe.process(frame)

        current_gesture = None
        if info is not None and "bbox" in info and "gesture" in info:
            draw_bbox_label(frame, info["bbox"], info["gesture"])
            current_gesture = info["gesture"]

        # 목표 제스처가 설정된 상태라면, 안내 + 매칭 체크
        if target is not None and start_time is not None:
            # 목표 제스처와 반응속도 정보 표시
            draw_reaction_info(
                frame,
                target=target,
                reaction_time=reaction_time,
                mode_text="MODE: REACTION",
            )

            # 학생 제스처가 타겟과 같아지면 반응속도 측정
            if current_gesture == target and reaction_time is None:
                reaction_time = time.time() - start_time
                print(
                    f"Target: {target}, Reaction Time: {reaction_time:.3f} sec"
                )

        else:
            # 목표가 없을 때도, 마지막 반응속도는 계속 보여줄 수 있음
            draw_reaction_info(
                frame,
                target=None,
                reaction_time=reaction_time,
                mode_text="MODE: REACTION",
            )

        cv2.imshow("RPS Game - Reaction Mode", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # R 키 → 새로운 목표 제스처 생성
        if key == ord("r"):
            target = random.choice(GESTURES)
            start_time = time.time()
            reaction_time = None
            print(f"New target gesture: {target}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
