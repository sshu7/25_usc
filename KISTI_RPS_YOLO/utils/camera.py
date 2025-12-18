import cv2

class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(index)
        # 성능을 위해 해상도 설정(환경에 맞게 조정)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
