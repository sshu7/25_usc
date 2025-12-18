# utils/visualization.py
"""
Visualization utilities for KISTI RPS YOLO game.

이 파일은 다음 기능을 제공합니다:
- 손 박스 및 제스처 라벨 그리기 (draw_bbox_label)
- 기본 HUD (YOU / CPU / RESULT) 표시 (draw_hud_basic / draw_hud)
- 전적(승/무/패) HUD 표시 (draw_hud_stats)
- 반응속도 모드용 정보 표시 (draw_reaction_info)

KISTI에서 제공한 YOLO 기반 파이프라인(utils.pipeline 등)은
그대로 사용하고, 화면에 그려주는 부분만 이 파일에서 담당합니다.
"""

import cv2
from typing import Tuple, Optional


# ---------- 색상 정의 (BGR) ----------
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0,  255,   0)
RED    = (0,   0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)


# ---------- 공통 유틸 ----------

def _ensure_int_bbox(bbox) -> Tuple[int, int, int, int]:
    """
    bbox 형식을 (x1, y1, x2, y2) 정수 튜플로 보정합니다.
    KISTI 파이프라인에서 bbox가 list/tuple 로 들어온다고 가정합니다.
    """
    if bbox is None:
        return 0, 0, 0, 0

    if len(bbox) != 4:
        # 예외 상황 대응: 길이가 4가 아니면 그냥 0,0,0,0
        return 0, 0, 0, 0

    x1, y1, x2, y2 = bbox
    try:
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
    except Exception:
        x1 = y1 = x2 = y2 = 0

    return x1, y1, x2, y2


# ---------- 1. 손 박스 + 라벨 ----------

def draw_bbox_label(frame, bbox, label: Optional[str] = None,
                    color: Tuple[int, int, int] = GREEN,
                    text_color: Tuple[int, int, int] = WHITE,
                    thickness: int = 2) -> None:
    """
    인식된 손 위치(bbox)에 박스를 그리고, 제스처 라벨을 표시합니다.
    bbox: (x1, y1, x2, y2)
    label: "rock", "paper", "scissors" 등
    """

    if frame is None or bbox is None:
        return

    x1, y1, x2, y2 = _ensure_int_bbox(bbox)

    # 박스 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if not label:
        return

    # 라벨 배경 박스
    label_text = str(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    # 텍스트 크기 계산
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, font, font_scale, font_thickness
    )

    # 텍스트 배경 위치
    text_x = x1
    text_y = max(y1 - 10, text_h + 10)  # 화면 위를 벗어나지 않도록

    cv2.rectangle(
        frame,
        (text_x, text_y - text_h - baseline),
        (text_x + text_w, text_y + baseline),
        color,
        thickness=-1,
    )

    # 텍스트 그리기
    cv2.putText(
        frame,
        label_text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        lineType=cv2.LINE_AA,
    )


# ---------- 2. 기본 HUD (YOU / CPU / RESULT) ----------

def draw_hud(frame,
             you: Optional[str],
             cpu: Optional[str],
             result: Optional[str]) -> None:
    """
    기존 코드와의 호환을 위해 남겨둔 기본 HUD 함수.
    아래 draw_hud_basic과 동일한 역할을 합니다.
    """
    draw_hud_basic(frame, you, cpu, result)


def draw_hud_basic(frame,
                   you: Optional[str],
                   cpu: Optional[str],
                   result: Optional[str]) -> None:
    """
    화면 상단에 간단한 HUD를 표시합니다.
    - YOU: 사용자의 현재 제스처
    - CPU: 컴퓨터의 제스처
    - RESULT: win / lose / draw
    """
    if frame is None:
        return

    h, w = frame.shape[:2]

    # 상단 HUD 영역 배경
    cv2.rectangle(frame, (0, 0), (w, 40), BLACK, thickness=-1)

    text = f"YOU: {you or '-'} | CPU: {cpu or '-'} | RESULT: {result or '-'}"
    cv2.putText(
        frame,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2,
        lineType=cv2.LINE_AA,
    )


# ---------- 3. 전적(승/무/패) HUD ----------

def draw_hud_stats(frame,
                   you: Optional[str],
                   cpu: Optional[str],
                   result: Optional[str],
                   total: int,
                   wins: int,
                   draws: int,
                   losses: int) -> None:
    """
    전적(총 판수 / 승 / 무 / 패)을 포함하는 HUD.
    run_game_stats.py 에서 사용합니다.
    """
    if frame is None:
        return

    h, w = frame.shape[:2]

    # HUD 영역(위쪽 80픽셀 정도) 배경
    cv2.rectangle(frame, (0, 0), (w, 80), BLACK, thickness=-1)

    line1 = f"YOU: {you or '-'} | CPU: {cpu or '-'} | RESULT: {result or '-'}"
    line2 = f"TOTAL: {total} | WIN: {wins} | DRAW: {draws} | LOSE: {losses}"

    cv2.putText(
        frame,
        line1,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        line2,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        YELLOW,
        2,
        lineType=cv2.LINE_AA,
    )


# ---------- 4. 반응속도 모드 정보 표시 ----------

def draw_reaction_info(frame,
                       target: Optional[str],
                       reaction_time: Optional[float],
                       mode_text: Optional[str] = None) -> None:
    """
    반응속도 모드에서 목표 제스처와 반응속도 정보를 표시합니다.
    - target: 현재 맞춰야 할 제스처 ("rock", "paper", "scissors")
    - reaction_time: 마지막 반응속도(sec)
    - mode_text: "MODE: REACTION" 처럼 상태를 알려주는 텍스트
    """
    if frame is None:
        return

    h, w = frame.shape[:2]

    # 모드 텍스트 (옵션)
    if mode_text:
        cv2.putText(
            frame,
            mode_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            CYAN,
            2,
            lineType=cv2.LINE_AA,
        )

    # 반응속도 표시
    if reaction_time is not None:
        cv2.putText(
            frame,
            f"RT: {reaction_time:.3f} s",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            GREEN,
            2,
            lineType=cv2.LINE_AA,
        )

    # 목표 제스처 안내
    if target is not None:
        cv2.putText(
            frame,
            f"SHOW: {target.upper()}!",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            YELLOW,
            3,
            lineType=cv2.LINE_AA,
        )
