from typing import Optional, Tuple 
import numpy as np
import cv2 
import mediapipe as mp

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils

class RPSPipeline: 
    def __init__(self, max_hands: int = 1, det_conf: float = 0.5, track_conf: float = 0.5):
        # 여기서 MediaPipe Hands 초기화
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.gesture_history = []  # 최근 몇 프레임의 결과 저장
        self.history_size = 5
    
    def classify_gesture(self, landmarks, hand_label: str = "Left") -> str:
        """개선된 rule-based 가위바위보 분류"""
        fingers = []
        
        # MediaPipe의 hand_label은 실제와 반대로 나옴 (카메라 미러링)
        actual_hand = "Left" if hand_label == "Right" else "Right"
        
        # 손바닥/손등 판단
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        mcp_middle = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        mcp_index = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        
        palm_vec1 = mcp_middle - wrist
        palm_vec2 = mcp_index - wrist
        palm_normal = np.cross(palm_vec1, palm_vec2)
        is_palm_facing = palm_normal[2] > 0  # 손바닥이 카메라를 향하는지
        
        # 엄지 판정 - 손바닥/손등에 따라 다르게 처리
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        if is_palm_facing:  # 손바닥이 보일 때
            if actual_hand == "Right":
                # 실제 오른손 손바닥: 엄지가 오른쪽으로 펼쳐짐
                thumb_open = thumb_tip.x > thumb_mcp.x
            else:
                # 실제 왼손 손바닥: 엄지가 오른쪽으로 펼쳐짐
                thumb_open = thumb_tip.x > thumb_mcp.x
        else:  # 손등이 보일 때
            if actual_hand == "Right":
                # 실제 오른손 손등: 엄지가 왼쪽으로 펼쳐짐
                thumb_open = thumb_tip.x < thumb_mcp.x
            else:
                # 실제 왼손 손등: 엄지가 왼쪽으로 펼쳐짐  
                thumb_open = thumb_tip.x < thumb_mcp.x
        
        fingers.append(thumb_open)
        
        # 나머지 손가락들 - Y축 기준 + 약간의 여유
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, pip)
        
        for tip_idx, pip_idx in finger_pairs:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp_idx = pip_idx - 1  # pip 바로 아래가 mcp
            mcp = landmarks[mcp_idx]
            
            # Y축 기준 판단 (tip이 mcp보다 충분히 위에 있으면 열림)
            y_diff = mcp.y - tip.y
            is_extended = y_diff > 0.02  # 임계값을 약간 여유있게
            
            fingers.append(is_extended)
        
        # 디버깅 출력
       # print(f"[DEBUG] MediaPipe: {hand_label}, Actual: {actual_hand}, Palm facing: {is_palm_facing}, Fingers: {fingers}")
        
        # 열린 손가락 개수로 간단하게 분류
        open_count = sum(fingers)
        
        if open_count <= 1:  # 0개 또는 1개만 열림
            return "rock"
        elif open_count >= 4:  # 4개 이상 열림
            return "paper"
        elif open_count == 2:
            # 2개가 열린 경우 - 주로 scissors
            if fingers[1] and fingers[2]:  # 검지 + 중지
                return "scissors"
            elif fingers[1] and fingers[3]:  # 검지 + 약지  
                return "scissors"
            elif fingers[2] and fingers[3]:  # 중지 + 약지
                return "scissors"
            else:
                return "scissors"  # 2개 열린 경우 scissors로 간주
        elif open_count == 3:
            # 3개 열린 경우
            if not fingers[0] and fingers[1] and fingers[2]:  # 엄지 닫고 검지+중지 열림
                return "scissors"
            else:
                return "paper"  # 나머지는 paper로 간주
        else:
            return "unknown"
    
    def smooth_gesture(self, current_gesture: str) -> str:
        """최근 몇 프레임의 결과를 이용해 안정화"""
        self.gesture_history.append(current_gesture)
        
        # 히스토리 크기 유지
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # unknown이 아닌 것들만 고려
        valid_gestures = [g for g in self.gesture_history if g != "unknown"]
        
        if not valid_gestures:
            return "unknown"
        
        # 최다 빈도 제스처 반환
        from collections import Counter
        gesture_counts = Counter(valid_gestures)
        most_common = gesture_counts.most_common(1)[0][0]
        
        # 최근 2프레임 중 하나라도 같으면 해당 제스처 사용
        recent_gestures = self.gesture_history[-2:]
        if most_common in recent_gestures:
            return most_common
        else:
            return current_gesture
    
    def infer_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # 첫 번째 손만 사용
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_label = "Left"
        if results.multi_handedness:
            hand_label = results.multi_handedness[0].classification[0].label
        
        # bounding box 계산
        h, w = frame_bgr.shape[:2]
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        
        # 제스처 분류 및 평활화
        raw_gesture = self.classify_gesture(hand_landmarks.landmark, hand_label)
        smoothed_gesture = self.smooth_gesture(raw_gesture)
        
        return {
            'box': (x1, y1, x2, y2),
            'det_conf': 1.0,
            'gesture': smoothed_gesture,
            'gesture_conf': 1.0,
            'hand_label': hand_label,
            'landmarks': hand_landmarks,
        }                             
