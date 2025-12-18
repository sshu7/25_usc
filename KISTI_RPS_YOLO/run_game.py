import argparse
import cv2
from utils.camera import Camera
from utils.visualization import draw_bbox_label, draw_hud
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
    
    print("Press 'q' to quit. Press 'space' to lock your move vs CPU.")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        info = pipe.infer_frame(frame)
        
        if info is not None:
            x1, y1, x2, y2 = info['box']
            draw_bbox_label(frame, info['box'], f"hand {info['det_conf']:.2f}")
            cv2.putText(frame, f"gesture: {info['gesture']} ({info['gesture_conf']:.2f})", 
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        draw_hud(frame, last_you, last_cpu, last_result)
        cv2.imshow("Rock-Paper-Scissors", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 스페이스바로 판정 트리거
            if info is not None:
                you = info['gesture']
                cpu = cpu_choice()
                result = decide(you, cpu)
                last_you, last_cpu, last_result = you, cpu, result
                print(f"You: {you}, CPU: {cpu}, Result: {result}")  # 디버깅용 출력 추가
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()