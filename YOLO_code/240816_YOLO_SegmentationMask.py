from ultralytics import YOLO
import cv2
import numpy as np
import logging


# 상수 Define
ESC_KEY = ord('q')          # 캠 종료 버튼
WEBCAM_INDEX = 2            # 사용하고자 하는 웹캠 장치의 인덱스
FRAME_WIDTH = 640           # 웹캠 프레임 너비
FRAME_HEIGHT = 480          # 웹캠 프레임 높이
CONFIDENCE_THRESHOLD = 0.7  # YOLO 모델의 신뢰도 임계값
DEFAULT_MODEL_PATH = '/home/beakhongha/YOLO_ARIS/train23/weights/best.pt'   # YOLO 모델의 경로

# 로깅 수준을 WARNING으로 설정하여 정보 메시지 비활성화
logging.getLogger("ultralytics").setLevel(logging.WARNING)


class YOLOMain:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, webcam_index=WEBCAM_INDEX, 
                 frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, conf=CONFIDENCE_THRESHOLD):
        """
        클래스 초기화 메서드.
        모델을 로드하고 웹캠을 초기화합니다.
        """
        self.model = YOLO(model_path)
        self.webcam = cv2.VideoCapture(webcam_index)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.conf = conf

        if not self.webcam.isOpened():
            raise Exception("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")
        
        # 객체 인식 바운딩 박스 및 마크스 색상 설정
        self.colors = {
            'cup': (0, 255, 0),
            'capsule': (0, 0, 255),
            'capsule_label': (255, 255, 0),
            'capsule_not_label': (0, 255, 255),
            'robot': (0, 165, 255),
            'human': (255, 0, 0),
            'cup_holder': (255, 255, 255)
        }


    def predict_on_image(self, img):
        """
        입력된 이미지에 대해 예측을 수행하는 메서드.
        YOLO 모델을 사용해 바운딩 박스, 마스크, 클래스, 신뢰도 점수를 반환합니다.

        :param img: 예측할 이미지
        :return: 바운딩 박스, 마스크, 클래스, 신뢰도 점수
        """
        result = self.model(img, conf=self.conf)[0]

        # Detection
        cls = result.boxes.cls.cpu().numpy()
        probs = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        # Segmentation
        masks = result.masks.data.cpu().numpy()
        
        return boxes, masks, cls, probs


    def overlay(self, image, mask, color, alpha=0.5):
        """
        이미지 위에 세그멘테이션 마스크를 오버레이하는 메서드.
        주어진 색상과 투명도를 사용하여 마스크를 원본 이미지에 결합합니다.

        :param image: 원본 이미지
        :param mask: 세그멘테이션 마스크
        :param color: 마스크를 표시할 색상
        :param alpha: 마스크와 원본 이미지의 혼합 비율
        :return: 마스크가 오버레이된 이미지
        """
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        try:
            mask_indices = mask > 0
            overlay_image = image.copy()
            overlay_image[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)
        except Exception as e:
            print(f"오버레이 처리 중 오류 발생: {e}")
            return image  # 오류 발생 시 원본 이미지를 반환
        
        return overlay_image


    def process_frame(self, frame):
        """
        웹캠 프레임을 처리하여 예측 결과를 오버레이하는 메서드.
        YOLO 모델을 사용하여 프레임에 대한 예측을 수행하고, 마스크와 바운딩 박스를 이미지에 표시합니다.

        :param frame: 웹캠에서 읽은 프레임
        :return: 마스크와 바운딩 박스가 오버레이된 프레임
        """
        # 현재 프레임 예측
        boxes, masks, cls, probs = self.predict_on_image(frame)

        # 원본 이미지에 마스크 오버레이 및 디텍션 박스 표시
        image_with_masks = np.copy(frame)

        # 각 객체에 대해 박스, 마스크 생성
        for box, mask, class_id, prob in zip(boxes, masks, cls, probs):
            label = self.model.names[int(class_id)]
            if label == 'hand':  # 'hand' 객체를 'human' 객체로 변경하여 출력
                label = 'human'
            color = self.colors.get(label, (255, 255, 255))

            image_with_masks = self.overlay(image_with_masks, mask, color, alpha=0.3)

            # 디텍션 박스 및 라벨 시각화
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_with_masks, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_with_masks, f'{label}: {prob:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image_with_masks
    

    def release_resources(self):
        """
        웹캠과 OpenCV 창을 해제하는 메서드.
        프로그램 종료 시 호출하여 자원을 안전하게 해제합니다.
        """
        self.webcam.release()
        cv2.destroyAllWindows()


    def run_yolo(self):
        """
        YOLO 모델을 실행하는 메서드.
        실시간으로 웹캠 영상을 처리하고, 예측 결과를 화면에 출력합니다.
        """
        # 카메라 작동
        while True:
            ret, frame = self.webcam.read() # 웹캠에서 프레임 읽기
            if not ret:                     # 프레임을 읽지 못한 경우 오류 메시지 출력, 루프 종료
                print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
                break
            
            # 웹캠 프레임을 처리하여 예측 결과를 오버레이
            image_with_masks = self.process_frame(frame)

            # 디텍션 박스와 마스크가 적용된 프레임 표시
            cv2.imshow("Webcam with Segmentation Masks and Detection Boxes", image_with_masks)

            # 종료 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                break

        # 자원 해제
        self.release_resources()


if __name__ == "__main__":
    yolomain = YOLOMain()
    yolomain.run_yolo()