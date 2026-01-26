import argparse
import os
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

# Hugging Face model checkpoint (PyTorch Lightning)
CKPT_URL = "https://huggingface.co/MichaelMM2000/animals10-resnet/resolve/main/resnet_animals10.ckpt"
CKPT_PATH_DEFAULT = "./weights/resnet_animals10.ckpt"

CLASSES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]


def ensure_ckpt(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return

    print(f"[INFO] Downloading checkpoint:\n  {CKPT_URL}\n  -> {path}")
    import urllib.request
    urllib.request.urlretrieve(CKPT_URL, path)
    print("[INFO] Download done.")


def build_model(device: str, ckpt_path: str) -> torch.nn.Module:
    """
    Load Lightning checkpoint and map to a plain torchvision ResNet18 classifier (10 classes).
    """
    ensure_ckpt(ckpt_path)

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning checkpoints usually have 'state_dict'
    state = ckpt.get("state_dict", ckpt)

    # Normalize key names (try common prefixes)
    cleaned = {}
    for k, v in state.items():
        kk = k
        for prefix in ("model.", "net.", "backbone.", "classifier."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        cleaned[kk] = v

    # Sometimes keys are like "resnet.conv1.weight" etc.
    # Try to strip a leading "resnet."
    cleaned2 = {}
    for k, v in cleaned.items():
        if k.startswith("resnet."):
            cleaned2[k[len("resnet."):]] = v
        else:
            cleaned2[k] = v

    missing, unexpected = model.load_state_dict(cleaned2, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.eval().to(device)
    return model


def preprocess_pil(pil: Image.Image) -> torch.Tensor:
    # ResNet18 standard preprocess
    # (ImageNet-style normalization works fine for this checkpoint too in practice)
    import torchvision.transforms as T
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil)


def to_tensor_bgr(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = preprocess_pil(pil).unsqueeze(0).to(device)
    return x


def topk_from_logits(logits: torch.Tensor, k: int) -> List[Tuple[int, float]]:
    probs = torch.softmax(logits, dim=1)
    vals, idxs = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
    idxs = idxs[0].detach().cpu().tolist()
    vals = vals[0].detach().cpu().tolist()
    return list(zip(idxs, vals))


def draw_text_block(frame, x, y, w, h, alpha=0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_topk(frame, items: List[Tuple[str, float]], origin=(20, 45), title="Top-K",
              font_scale=1.1, thickness=3, max_y=260):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, th), base = cv2.getTextSize("A", font, font_scale, thickness)
    line_h = th + base + 10

    cv2.putText(frame, title, (x, y), font, font_scale * 1.1, (255, 255, 255),
                thickness + 1, cv2.LINE_AA)
    y += line_h

    for i, (name, p) in enumerate(items, 1):
        if y > max_y:
            cv2.putText(frame, "...", (x, y), font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
            break
        cv2.putText(frame, f"{i}. {name}  {p*100:5.1f}%", (x, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h


def draw_bar(frame, items: List[Tuple[str, float]], x=20, y=360, w=520, bar_h=28, gap=12):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "FREEZE: Detail Top-K", (x, y - 25), font, 0.95, (255, 255, 255), 3, cv2.LINE_AA)

    for i, (name, p) in enumerate(items):
        yy = y + i * (bar_h + gap)
        cv2.rectangle(frame, (x, yy), (x + w, yy + bar_h), (70, 70, 70), -1)
        fill = int(w * max(0.0, min(1.0, p)))
        cv2.rectangle(frame, (x, yy), (x + fill, yy + bar_h), (200, 200, 200), -1)
        cv2.putText(frame, f"{name:<10} {p*100:5.1f}%", (x + w + 20, yy + bar_h - 5),
                    font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def status_line(frame, text: str):
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--win_w", type=int, default=1280)
    ap.add_argument("--win_h", type=int, default=720)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--detail_topk", type=int, default=10)
    ap.add_argument("--infer_interval", type=float, default=0.18)
    ap.add_argument("--ckpt", type=str, default=CKPT_PATH_DEFAULT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device, args.ckpt)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed. Check /dev/video0 and permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win = "Animals-10 (LIVE/Freeze)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.win_w, args.win_h)

    mode = "LIVE"
    freeze_frame: Optional[np.ndarray] = None
    freeze_items = None

    last_infer = 0.0
    live_items: List[Tuple[str, float]] = []

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    while True:
        if mode == "LIVE":
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            fps_n += 1
            if now - fps_t0 >= 1.0:
                fps = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0

            if now - last_infer >= args.infer_interval:
                last_infer = now
                with torch.inference_mode():
                    x = to_tensor_bgr(frame, device)
                    logits = model(x)
                idx_prob = topk_from_logits(logits, k=max(args.detail_topk, args.topk))
                live_items = [(CLASSES[i], p) for i, p in idx_prob]

            draw_text_block(frame, 5, 5, 650, 310, alpha=0.45)
            draw_topk(frame, live_items[:args.topk], title="LIVE: Animal Similarity (Animals-10)")
            status_line(frame, f"Mode: LIVE | FPS: {fps:.1f} | SPACE=Freeze  R=Resume  Q=Quit")
            cv2.imshow(win, frame)

        else:
            frame = freeze_frame.copy()
            draw_text_block(frame, 5, 5, 1250, 680, alpha=0.40)
            draw_topk(frame, freeze_items[:args.topk], title="FREEZE: Animal Similarity (Animals-10)", font_scale=1.2, max_y=260)
            draw_bar(frame, freeze_items[:args.detail_topk])
            status_line(frame, "Mode: FREEZE | R=Resume  Q=Quit")
            cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key == ord(' '):
            if mode == "LIVE":
                freeze_frame = frame.copy()
                freeze_items = list(live_items) if live_items else []
                mode = "FREEZE"
        if key in (ord('r'), ord('R')):
            mode = "LIVE"
            freeze_frame = None
            freeze_items = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

