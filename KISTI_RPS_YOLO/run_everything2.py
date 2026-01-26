import argparse
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision import models
from PIL import Image


# -----------------------------
# Animal filtering helpers
# -----------------------------
ANIMAL_KEYWORDS = {
    # general
    "dog", "cat", "kitten", "puppy",
    "fox", "wolf", "coyote", "hyena",
    "lion", "tiger", "leopard", "cheetah", "panther", "jaguar",
    "bear", "panda",
    "horse", "zebra", "donkey", "mule",
    "cow", "ox", "bull", "bison", "buffalo",
    "sheep", "goat", "ram",
    "pig", "boar", "hog",
    "deer", "elk", "moose", "reindeer", "antelope",
    "camel", "llama", "alpaca",
    "rabbit", "hare",
    "mouse", "rat", "hamster", "guinea pig",
    "squirrel", "chipmunk",
    "monkey", "ape", "chimpanzee", "gorilla", "orangutan",
    "elephant", "hippopotamus", "rhino", "rhinoceros", "giraffe",
    "kangaroo", "koala",
    # birds
    "bird", "eagle", "hawk", "falcon", "owl", "sparrow", "robin", "crow",
    "parrot", "penguin", "duck", "goose", "swan", "chicken", "hen", "rooster",
    "turkey", "peacock",
    # reptiles/amphibians
    "snake", "lizard", "crocodile", "alligator", "turtle", "tortoise",
    "frog", "toad",
    # fish/sea
    "fish", "shark", "whale", "dolphin", "seal", "otter", "walrus",
    "octopus", "squid", "crab", "lobster", "starfish", "jellyfish",
    # insects
    "butterfly", "moth", "bee", "wasp", "ant", "spider", "beetle", "ladybug", "mosquito"
}

def is_animal_label(label: str) -> bool:
    """Heuristic: ImageNet label string contains animal-ish keywords."""
    s = label.lower()
    # some labels contain commas, e.g., "tabby, tabby cat"
    # split and test tokens and phrases
    parts = [p.strip() for p in s.replace("-", " ").split(",")]
    for p in parts:
        tokens = p.split()
        # direct match any token
        for t in tokens:
            if t in ANIMAL_KEYWORDS:
                return True
        # also match whole phrase (rare but harmless)
        if p in ANIMAL_KEYWORDS:
            return True
    return False


# -----------------------------
# Model and preprocessing
# -----------------------------
def get_model_and_preprocess(device: str):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.eval().to(device)

    categories = weights.meta.get("categories", None)
    if categories is None:
        categories = [f"class_{i}" for i in range(1000)]

    preprocess = weights.transforms()  # expects PIL or torch Tensor
    return model, preprocess, categories


def to_tensor_bgr(frame_bgr: np.ndarray, preprocess, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    x = preprocess(pil_img).unsqueeze(0).to(device)
    return x


def softmax_topk(logits: torch.Tensor, k: int) -> List[Tuple[int, float]]:
    probs = torch.softmax(logits, dim=1)
    vals, idxs = torch.topk(probs, k=k, dim=1)
    idxs = idxs[0].detach().cpu().numpy().tolist()
    vals = vals[0].detach().cpu().numpy().tolist()
    return list(zip(idxs, vals))


# -----------------------------
# UI helpers
# -----------------------------
def draw_text_block(frame, x, y, w, h, alpha=0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_topk_limited(
    frame_bgr: np.ndarray,
    topk: List[Tuple[str, float]],
    origin=(20, 45),
    max_y=300,
    title="Top-K",
    font_scale=1.05,
    thickness=3,
):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, th), base = cv2.getTextSize("A", font, font_scale, thickness)
    line_h = th + base + 10

    cv2.putText(frame_bgr, title, (x, y), font, font_scale * 1.10, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    y += line_h

    for i, (name, p) in enumerate(topk, start=1):
        if y > max_y:
            cv2.putText(frame_bgr, "...", (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            break
        text = f"{i}. {name}  {p*100:.1f}%"
        cv2.putText(frame_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h


def draw_bar_chart(
    frame_bgr: np.ndarray,
    topk: List[Tuple[str, float]],
    x=20,
    y=170,
    w=520,
    bar_h=26,
    gap=12,
    title="Detail Top-K (Animal-only)",
    font_scale=0.75,
    thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_bgr, title, (x, y - 25), font, 0.9, (255, 255, 255), 3, cv2.LINE_AA)

    for i, (name, p) in enumerate(topk):
        yy = y + i * (bar_h + gap)
        # background
        cv2.rectangle(frame_bgr, (x, yy), (x + w, yy + bar_h), (70, 70, 70), -1)
        fill = int(w * max(0.0, min(1.0, p)))
        cv2.rectangle(frame_bgr, (x, yy), (x + fill, yy + bar_h), (200, 200, 200), -1)

        label = f"{name[:28]:<28} {p*100:5.1f}%"
        cv2.putText(
            frame_bgr,
            label,
            (x + w + 20, yy + bar_h - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def put_status_line(frame_bgr: np.ndarray, text: str, font_scale=0.9, thickness=2):
    h = frame_bgr.shape[0]
    cv2.putText(
        frame_bgr,
        text,
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# -----------------------------
# Core logic: animal-only topk
# -----------------------------
def build_animal_topk(labels: List[str], idx_prob: List[Tuple[int, float]], want_k: int) -> List[Tuple[str, float]]:
    out = []
    for idx, p in idx_prob:
        name = labels[idx]
        if is_animal_label(name):
            out.append((name, p))
        if len(out) >= want_k:
            break
    return out


def compute_animal_predictions(
    model,
    preprocess,
    labels,
    frame_bgr: np.ndarray,
    device: str,
    search_k: int,
    animal_topk: int,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], float]:
    """
    Returns:
      animal_top: animal-only topk (len <= animal_topk)
      general_top: general topk (for fallback / debug)
      animal_mass: sum of probabilities of animal entries in the top search_k list
    """
    with torch.inference_mode():
        x = to_tensor_bgr(frame_bgr, preprocess, device)
        logits = model(x)

    # general candidates
    idx_prob = softmax_topk(logits, k=search_k)
    general_top = [(labels[i], p) for i, p in idx_prob[:min(5, len(idx_prob))]]

    # animal-only from candidates
    animal_top = build_animal_topk(labels, idx_prob, want_k=animal_topk)

    # animal probability mass within top search_k
    animal_mass = 0.0
    for i, p in idx_prob:
        if is_animal_label(labels[i]):
            animal_mass += p

    return animal_top, general_top, animal_mass


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--win_w", type=int, default=1280)
    ap.add_argument("--win_h", type=int, default=720)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--detail_topk", type=int, default=10)
    ap.add_argument("--search_k", type=int, default=30, help="how many general classes to search before filtering to animals")
    ap.add_argument("--animal_threshold", type=float, default=0.15, help="if animal prob mass is below this, show Non-animal")
    ap.add_argument("--infer_interval", type=float, default=0.18, help="seconds between inferences in LIVE")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, labels = get_model_and_preprocess(device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed. Check /dev/video0 and permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win = "Animal Similarity (LIVE/Freeze)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.win_w, args.win_h)

    mode = "LIVE"
    freeze_frame: Optional[np.ndarray] = None
    freeze_payload = None  # (animal_top, animal_detail, general_top, animal_mass)

    last_infer_t = 0.0
    live_animal_top: List[Tuple[str, float]] = []
    live_general_top: List[Tuple[str, float]] = []
    live_animal_mass = 0.0

    # FPS
    fps_t0 = time.time()
    fps_counter = 0
    fps_value = 0.0

    while True:
        if mode == "LIVE":
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            fps_counter += 1
            if now - fps_t0 >= 1.0:
                fps_value = fps_counter / (now - fps_t0)
                fps_t0 = now
                fps_counter = 0

            if now - last_infer_t >= args.infer_interval:
                last_infer_t = now
                animal_top, general_top, animal_mass = compute_animal_predictions(
                    model, preprocess, labels, frame, device,
                    search_k=args.search_k,
                    animal_topk=args.topk
                )
                live_animal_top = animal_top
                live_general_top = general_top
                live_animal_mass = animal_mass

            # UI blocks
            draw_text_block(frame, x=5, y=5, w=650, h=310, alpha=0.45)

            if live_animal_mass < args.animal_threshold or len(live_animal_top) == 0:
                # non-animal
                draw_topk_limited(
                    frame,
                    live_general_top,
                    origin=(20, 45),
                    max_y=260,
                    title="LIVE: Non-animal scene (Top General)",
                    font_scale=1.05,
                    thickness=3,
                )
            else:
                draw_topk_limited(
                    frame,
                    live_animal_top,
                    origin=(20, 45),
                    max_y=260,
                    title="LIVE: Animal Similarity (Top-K)",
                    font_scale=1.10,
                    thickness=3,
                )

            put_status_line(
                frame,
                f"Mode: LIVE | FPS: {fps_value:.1f} | SPACE=Freeze  R=Resume  Q=Quit",
                font_scale=0.9,
                thickness=2,
            )
            cv2.imshow(win, frame)

        else:
            # FREEZE
            frame = freeze_frame.copy()
            animal_top, animal_detail, general_top, animal_mass = freeze_payload

            draw_text_block(frame, x=5, y=5, w=1250, h=680, alpha=0.40)

            if animal_mass < args.animal_threshold or len(animal_top) == 0:
                draw_topk_limited(
                    frame,
                    general_top,
                    origin=(20, 45),
                    max_y=260,
                    title="FREEZE: Non-animal scene (Top General)",
                    font_scale=1.15,
                    thickness=3,
                )
                cv2.putText(
                    frame,
                    f"Animal probability is low (mass={animal_mass:.2f}). Try showing an animal image on a phone.",
                    (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                draw_topk_limited(
                    frame,
                    animal_top,
                    origin=(20, 45),
                    max_y=260,
                    title="FREEZE: Animal Similarity (Top-K)",
                    font_scale=1.20,
                    thickness=3,
                )
                draw_bar_chart(
                    frame,
                    animal_detail,
                    x=20,
                    y=360,
                    w=520,
                    bar_h=28,
                    gap=12,
                    title="FREEZE: Detail (Animal-only Top-K)",
                    font_scale=0.80,
                    thickness=2,
                )

            put_status_line(frame, "Mode: FREEZE | R=Resume  Q=Quit", font_scale=0.9, thickness=2)
            cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

        if key == ord(' '):  # freeze
            if mode == "LIVE":
                freeze_frame = frame.copy()

                # compute detail on frozen frame
                animal_top, general_top, animal_mass = compute_animal_predictions(
                    model, preprocess, labels, freeze_frame, device,
                    search_k=args.search_k,
                    animal_topk=args.topk
                )
                # for detail, compute animal-only topk with larger desired k
                # reuse compute but ask for detail_topk
                animal_detail, _, _ = compute_animal_predictions(
                    model, preprocess, labels, freeze_frame, device,
                    search_k=args.search_k,
                    animal_topk=args.detail_topk
                )
                freeze_payload = (animal_top, animal_detail, general_top, animal_mass)
                mode = "FREEZE"

        if key in (ord('r'), ord('R')):
            mode = "LIVE"
            freeze_frame = None
            freeze_payload = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

