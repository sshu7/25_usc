import argparse
import cv2

from utils.camera import Camera
from utils.visualization import draw_bbox_label, draw_hud_stats
from utils.rps_logic import cpu_choice, decide
from detect_and_classify import RPSPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cam = Camera(args.camera)
    pipe = RPSPipeline()

    last_you = None
    last_cpu = None
    last_result = None

    total = 0
    wins = 0
    draws = 0
    losses = 0

    print("=== STATS MODE ===")
    print("Press 'q' to quit. Press 'space' to lock your move vs CPU and update stats.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        info = pipe.infer_frame(frame)

        # run_basic과 동일한 방식으로 bbox/gesture 출력
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

        # 전적 HUD 표시(visualization.py에 이미 존재) :contentReference[oaicite:2]{index=2}
        draw_hud_stats(frame, last_you, last_cpu, last_result, total, wins, draws, losses)

        cv2.imshow("Rock-Paper-Scissors (STATS)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):  # SPACE: 판정 + 전적 업데이트
            if info is not None:
                you = info["gesture"]
                cpu = cpu_choice()
                result = decide(you, cpu)  # win/lose/draw :contentReference[oaicite:3]{index=3}

                last_you, last_cpu, last_result = you, cpu, result

                total += 1
                if result == "win":
                    wins += 1
                elif result == "draw":
                    draws += 1
                else:
                    losses += 1

                print(f"[{total}] YOU={you}, CPU={cpu}, RESULT={result} | W/D/L={wins}/{draws}/{losses}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''
########### old_codes ##############

# run_game_stats.py
import cv2
import argparse

from utils.pipeline import RPSPipeline
from utils.rps_logic import cpu_choice, decide
from utils.visualization import draw_bbox_label, draw_hud_stats
from utils.camera import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    cam = Camera(args.camera)
    pipe = RPSPipeline()

    last_you = None
    last_cpu = None
    last_result = None

    total = 0
    wins = 0
    draws = 0
    losses = 0

    print("=== RPS Game - Stats Mode ===")
    print("  - SPACE: 가위바위보 한 판 진행")
    print("  - Q: 종료")
    print("==============================")

    while True:
        frame = cam.read()
        if frame is None:
            print("❗ 카메라 프레임을 읽지 못했습니다. 종료합니다.")
            break

        # YOLO + 제스처 판별
        frame, info = pipe.process(frame)

        if info is not None and "bbox" in info and "gesture" in info:
            draw_bbox_label(frame, info["bbox"], info["gesture"])

        # HUD: 전적 포함
        draw_hud_stats(
            frame,
            you=last_you,
            cpu=last_cpu,
            result=last_result,
            total=total,
            wins=wins,
            draws=draws,
            losses=losses,
        )

        cv2.imshow("RPS Game - Stats Mode", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # SPACE → 한 판 진행
        if key == ord(" "):
            if info is not None and "gesture" in info:
                you = info["gesture"]
                cpu = cpu_choice()
                result = decide(you, cpu)

                last_you, last_cpu, last_result = you, cpu, result

                total += 1
                if result == "win":
                    wins += 1
                elif result == "draw":
                    draws += 1
                elif result == "lose":
                    losses += 1

                print(
                    f"You: {you}, CPU: {cpu}, Result: {result} "
                    f"(Total: {total}, W/D/L = {wins}/{draws}/{losses})"
                )
            else:
                print("손 제스처가 인식되지 않았습니다. 화면에 손을 잘 보이게 해 주세요.")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''
