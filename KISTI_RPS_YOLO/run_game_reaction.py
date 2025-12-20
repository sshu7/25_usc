import argparse
import cv2
import time
import random

from utils.camera import Camera
from utils.visualization import draw_bbox_label, draw_reaction_info
from utils.rps_logic import CHOICES
from detect_and_classify import RPSPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cam = Camera(args.camera)
    pipe = RPSPipeline()

    target = random.choice(CHOICES)
    start_t = None
    last_rt = None
    best_rt = None

    print("=== REACTION MODE ===")
    print("Press 'q' to quit.")
    print("Press 'r' to start a round (timer starts). Show the target gesture ASAP!")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        info = pipe.infer_frame(frame)

        # 박스/제스처 출력(= run_basic 동일 패턴) :contentReference[oaicite:4]{index=4}
        if info is not None:
            x1, y1, x2, y2 = info["box"]
            draw_bbox_label(frame, info["box"], f"hand {info['det_conf']:.2f}")
            cv2.putText(
                frame,
                f"gesture: {info['gesture']} ({info['gesture_conf']:.2f})",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # 상단 안내 + 목표 + 반응속도 표시(visualization.py에 이미 존재) :contentReference[oaicite:5]{index=5}
        draw_reaction_info(frame, target=target, reaction_time=last_rt, mode_text="MODE: REACTION  (R=start, Q=quit)")

        if best_rt is not None:
            cv2.putText(
                frame,
                f"BEST: {best_rt:.3f} s",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )

        cv2.imshow("Rock-Paper-Scissors (REACTION)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            start_t = time.time()
            last_rt = None
            # 시작할 때마다 목표를 새로 뽑아도 재미있음
            target = random.choice(CHOICES)

        # 타이머가 시작된 상태에서 목표 제스처를 맞추면 반응시간 측정
        if start_t is not None and info is not None:
            if info["gesture"] == target:
                last_rt = time.time() - start_t
                if best_rt is None or last_rt < best_rt:
                    best_rt = last_rt

                print(f"TARGET={target}  RT={last_rt:.3f}s  BEST={best_rt:.3f}s")

                # 다음 라운드로 자동 진행(학생들이 계속 도전 가능)
                start_t = time.time()
                target = random.choice(CHOICES)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''
####### old codes #######

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
'''

