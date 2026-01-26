from PIL import Image
import argparse
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models

# -----------------------------
# Utilities
# -----------------------------
def load_imagenet_labels() -> List[str]:
    """
    ImageNet label list (1000 classes).
    torchvision 자체에 완전한 라벨 텍스트를 공식적으로 내장해두진 않아,
    보통은 labels.txt를 포함시키거나, 모델 weights meta에 있는 categories를 사용합니다.
    최신 torchvision에서는 weights.meta["categories"]에 들어있습니다.
    """
    return []


def get_model_and_labels(device: str):
    """
    MobileNet 계열로 빠른 추론. torchvision weights meta에서 categories 가져오기.
    """
    # mobilenet_v3_small이 Nano에서 비교적 가볍습니다.
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.eval()

    # labels
    categories = weights.meta.get("categories", None)
    if categories is None:
        # fallback: 빈 라벨 (실제 운영에서는 labels 파일을 포함시키는 것을 권장)
        categories = [f"class_{i}" for i in range(1000)]

    # transforms
    preprocess = weights.transforms()

    model.to(device)
    return model, preprocess, categories


def softmax_topk(logits: torch.Tensor, k: int) -> List[Tuple[int, float]]:
    probs = torch.softmax(logits, dim=1)
    vals, idxs = torch.topk(probs, k=k, dim=1)
    idxs = idxs[0].detach().cpu().numpy().tolist()
    vals = vals[0].detach().cpu().numpy().tolist()
    return list(zip(idxs, vals))


def draw_topk(
    frame_bgr: np.ndarray,
    topk: List[Tuple[str, float]],
    origin=(10, 30),
    line_h=28,
    title="Top-K Predictions"
):
    x, y = origin
    cv2.putText(frame_bgr, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_h

    for i, (name, p) in enumerate(topk, start=1):
        text = f"{i}. {name}  {p*100:.1f}%"
        cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h


def draw_bar_chart(
    frame_bgr: np.ndarray,
    topk: List[Tuple[str, float]],
    x=10,
    y=120,
    w=360,
    bar_h=18,
    gap=10,
    title="Detail (Bar Chart)"
):
    cv2.putText(frame_bgr, title, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (name, p) in enumerate(topk):
        yy = y + i * (bar_h + gap)
        # background bar
        cv2.rectangle(frame_bgr, (x, yy), (x + w, yy + bar_h), (60, 60, 60), -1)
        # filled bar
        fill = int(w * max(0.0, min(1.0, p)))
        cv2.rectangle(frame_bgr, (x, yy), (x + fill, yy + bar_h), (200, 200, 200), -1)
        # label
        label = f"{name[:28]:<28} {p*100:5.1f}%"
        cv2.putText(frame_bgr, label, (x + w + 15, yy + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def detect_faces(frame_bgr: np.ndarray, face_cascade) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces if faces is not None else []


def draw_faces(frame_bgr: np.ndarray, faces) -> None:
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 255, 255), 2)


def to_tensor_bgr(frame_bgr, preprocess, device: str) -> torch.Tensor:
    # torchvision preprocess는 PIL/torch tensor를 기대하므로 BGR->RGB 후 변환
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    img = preprocess(pil_img)  # 이미 tensor로 변환까지 포함됨
    img = img.unsqueeze(0).to(device)
    return img


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--topk", type=int, default=3, help="live overlay top-k")
    ap.add_argument("--detail_topk", type=int, default=10, help="freeze detail top-k")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--use_faces", action="store_true", help="draw face boxes (optional)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, labels = get_model_and_labels(device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed. Check /dev/video0 and permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    face_cascade = None
    if args.use_faces:
        # OpenCV 기본 haarcascade 사용 (설치된 OpenCV에 포함된 경로)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(haar_path)

    mode = "LIVE"        # LIVE or FREEZE
    freeze_frame = None
    freeze_result = None  # (topk_live, topk_detail, faces)
    last_infer_t = 0.0
    infer_interval = 0.15  # 150ms마다 추론(성능 맞춰 조절)
    live_topk_cache = []

    fps_t0 = time.time()
    fps_counter = 0
    fps_value = 0.0

    win = "Animal Similarity (LIVE/Freeze)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if mode == "LIVE":
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()

            # FPS 계산
            fps_counter += 1
            if now - fps_t0 >= 1.0:
                fps_value = fps_counter / (now - fps_t0)
                fps_t0 = now
                fps_counter = 0

            # 일정 주기로만 추론 (Jetson 성능 고려)
            if now - last_infer_t >= infer_interval:
                last_infer_t = now
                with torch.inference_mode():
                    x = to_tensor_bgr(frame, preprocess, device)
                    logits = model(x)
                    topk_idx = softmax_topk(logits, args.topk)

                live_topk_cache = [(labels[i], p) for i, p in topk_idx]

            # (선택) 얼굴 검출
            faces = []
            if face_cascade is not None:
                faces = detect_faces(frame, face_cascade)
                draw_faces(frame, faces)

            # 라이브 오버레이
            draw_topk(frame, live_topk_cache, origin=(10, 30), title="LIVE: Top-K (Most Similar)")
            cv2.putText(frame, f"Mode: LIVE  |  FPS: {fps_value:.1f}  |  SPACE=Freeze  R=Resume  Q=Quit",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, frame)

        else:
            # FREEZE mode
            frame = freeze_frame.copy()

            topk_live, topk_detail, faces = freeze_result

            # 표시 강화: top-3 + top-10 bar
            if faces is not None:
                draw_faces(frame, faces)

            draw_topk(frame, topk_live, origin=(10, 30), title="FREEZE: Top-K (Quick)")
            draw_bar_chart(frame, topk_detail, x=10, y=140, w=300, title="FREEZE: Detail Top-K (Bar)")

            cv2.putText(frame, "Mode: FREEZE  |  R=Resume  Q=Quit",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        if key == ord(' '):  # SPACE -> freeze
            if mode == "LIVE":
                # 현재 프레임을 정지하고, 더 자세한 분석을 한 번 수행
                freeze_frame = frame.copy()

                faces = None
                if face_cascade is not None:
                    faces = detect_faces(freeze_frame, face_cascade)

                with torch.inference_mode():
                    x = to_tensor_bgr(freeze_frame, preprocess, device)
                    logits = model(x)
                    topk_idx_live = softmax_topk(logits, args.topk)
                    topk_idx_detail = softmax_topk(logits, args.detail_topk)

                topk_live = [(labels[i], p) for i, p in topk_idx_live]
                topk_detail = [(labels[i], p) for i, p in topk_idx_detail]

                freeze_result = (topk_live, topk_detail, faces)
                mode = "FREEZE"

        if key in (ord('r'), ord('R')):
            mode = "LIVE"
            freeze_frame = None
            freeze_result = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

