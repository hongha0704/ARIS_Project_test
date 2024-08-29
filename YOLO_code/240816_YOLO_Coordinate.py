import numpy as np
import cv2 as cv
from ultralytics import YOLO
import logging

# 로깅 수준을 WARNING으로 설정하여 정보 메시지 비활성화
logging.getLogger("ultralytics").setLevel(logging.WARNING)

class YOLOMain:
    def __init__(self):
        """
        YOLOMain 클래스 초기화 메서드.
        카메라 좌표계와 로봇 좌표계 간의 호모그래피 변환 행렬을 계산합니다.
        """
        self.camera_points = np.array([
            [247.0, 121.0], [306.0, 107.0], [358.0, 94.0], [238.0, 79.0], [290.0, 66.0], [342.0, 52.0]
        ], dtype=np.float32)
        
        self.robot_points = np.array([
            [116.3, -424.9], [17.4, -456.5], [-73.2, -484.2], [140.1, -518.5], [45.6, -548.1], [-47.5, -580.8]
        ], dtype=np.float32)
        
        self.H = self.compute_homography_matrix()

    def compute_homography_matrix(self):
        """
        호모그래피 변환 행렬을 계산하는 메서드.
        카메라 좌표와 로봇 좌표를 기반으로 호모그래피 행렬을 계산합니다.
        """
        H, _ = cv.findHomography(self.camera_points, self.robot_points)
        print("호모그래피 변환 행렬 H:\n", H)
        return H

    def transform_to_robot_coordinates(self, image_points):
        """
        이미지 좌표를 로봇 좌표계로 변환하는 메서드.
        주어진 이미지 좌표를 로봇 좌표계로 변환합니다.

        :param image_points: 이미지 좌표 [x, y]
        :return: 로봇 좌표계로 변환된 좌표 [x, y]
        """
        camera_coords = np.array([[image_points]], dtype=np.float32)
        robot_coords = cv.perspectiveTransform(camera_coords, self.H)
        return [round(float(coord), 1) for coord in robot_coords[0][0]]

    def display_detection(self, frame, box, robot_coords, score):
        """
        탐지된 객체의 정보를 화면에 표시하는 메서드.
        바운딩 박스, 중심 좌표, 신뢰도를 화면에 그립니다.

        :param frame: 현재 프레임
        :param box: 객체의 바운딩 박스 좌표 [x1, y1, x2, y2]
        :param robot_coords: 로봇 좌표계로 변환된 중심 좌표 [x, y]
        :param score: 탐지 신뢰도
        """
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv.putText(frame, f'cup_holder Coord: ({robot_coords[0]:.2f}, {robot_coords[1]:.2f})', 
                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv.putText(frame, f'Confidence: {score:.2f}', 
        #            (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    """
    YOLOMain 클래스의 인스턴스를 생성하고, 웹캠을 통해 실시간 객체 탐지를 수행하는 메서드.
    'cup_holder' 객체를 탐지하여 화면에 바운딩 박스와 로봇 좌표를 표시합니다.
    """
    transformer = YOLOMain()

    cap = cv.VideoCapture(2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # YOLO 모델 로드
    model = YOLO('/home/beakhongha/YOLO_ARIS/train23/weights/best.pt')

    # 'cup_holder' 클래스의 인덱스 찾기
    cup_holder_index = next((idx for idx, name in model.names.items() if name == 'cup_holder'), None)

    if cup_holder_index is None:
        print("'cup_holder' 클래스가 모델에서 발견되지 않았습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # YOLO 모델을 사용하여 객체 탐지
        results = model(frame, conf=0.2)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):
                if int(cls) == cup_holder_index:
                    robot_coords = transformer.transform_to_robot_coordinates([
                        (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    ])
                    print("카메라 좌표계 좌표:", [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2])
                    print("로봇 좌표계 좌표:", robot_coords)
                    transformer.display_detection(frame, box, robot_coords, score)

        cv.imshow('Webcam', frame)
        print("")

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
