import numpy as np
import cv2 as cv
from ultralytics import YOLO

class CameraRobotTransformer:
    def __init__(self):
        # (카메라 좌표계와 로봇 좌표계에서 각각 측정된 좌표)
        self.camera_points =  np.array([
            [482, 211],  # 기준점 1의 카메라 좌표
            [487, 268],  # 기준점 2의 카메라 좌표
            [501, 327],  # 기준점 3의 카메라 좌표
            [425, 211],  # 기준점 4의 카메라 좌표
            [428, 268],  # 기준점 5의 카메라 좌표
            [429, 325],  # 기준점 6의 카메라 좌표
            [363, 212],  # 기준점 7의 카메라 좌표
            [366, 267],  # 기준점 8의 카메라 좌표
            [364, 325],  # 기준점 9의 카메라 좌표
            [113, 206],  # 기준점 10의 카메라 좌표
            [179, 208],  # 기준점 11의 카메라 좌표
            [242, 205],  # 기준점 12의 카메라 좌표
            [236, 328],  # 기준점 13의 카메라 좌표
            [232, 265],  # 기준점 14의 카메라 좌표
            ], dtype=np.float32)
        
        self.robot_points = np.array([
            [-300, -100],   # 기준점 1의 로봇 좌표
            [-300, 0],      # 기준점 2의 로봇 좌표
            [-300, 100],    # 기준점 3의 로봇 좌표
            [-200, -100],   # 기준점 4의 로봇 좌표
            [-200, 0],      # 기준점 5의 로봇 좌표
            [-200, 100],    # 기준점 6의 로봇 좌표
            [-100, -100],   # 기준점 7의 로봇 좌표
            [-100, 0],      # 기준점 8의 로봇 좌표
            [-100, 100],    # 기준점 9의 로봇 좌표
            [300, -100],    # 기준점 10의 로봇 좌표
            [200,-100],     # 기준점 11의 로봇 좌표
            [100, -100],    # 기준점 12의 로봇 좌표
            [100,100],      # 기준점 13의 로봇 좌표
            [100,0],        # 기준점 14의 로봇 좌표
            ], dtype=np.float32)
        
        self.H = self.compute_homography_matrix()

    def compute_homography_matrix(self):
        H, _ = cv.findHomography(self.camera_points, self.robot_points)
        print("호모그래피 변환 행렬 H:\n", H)
        return H

    def transform_to_robot_coordinates(self, image_points):
        camera_coords = np.array([image_points], dtype=np.float32)
        camera_coords = np.array([camera_coords])
        robot_coords = cv.perspectiveTransform(camera_coords, self.H)
        # 좌표를 소수점 한 자리로 반올림
        robot_coords = [round(float(coord), 1) for coord in robot_coords[0][0]]
        return robot_coords

def main():
    transformer = CameraRobotTransformer()

    cap = cv.VideoCapture(2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # YOLO 모델 로드
    model = YOLO('/home/beakhongha/YOLO_ARIS/train14/weights/best.pt')

    # 'capsule' 클래스의 인덱스 찾기
    capsule_index = None
    for idx, name in model.names.items():
        if name == 'capsule':
            capsule_index = idx
            break

    if capsule_index is None:
        print("'capsule' 클래스가 모델에서 발견되지 않았습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # YOLO 모델을 사용하여 객체 탐지
        results = model(frame, conf = 0.2)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):
                if int(cls) == capsule_index:
                    x1, y1, x2, y2 = map(int, box)
                    center_x_pixel = (x1 + x2) // 2
                    center_y_pixel = (y1 + y2) // 2

                    # 로봇 좌표계로 변환
                    robot_coords = transformer.transform_to_robot_coordinates([center_x_pixel, center_y_pixel])

                    # 결과 출력
                    print("카메라 좌표계 좌표:", [center_x_pixel, center_y_pixel])
                    print("로봇 좌표계 좌표:", robot_coords)

                    # 화면에 바운딩 박스와 중심 좌표 및 신뢰도 표시
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.circle(frame, (center_x_pixel, center_y_pixel), 5, (0, 255, 0), -1)
                    cv.putText(frame, f'Capsule Coord: ({robot_coords[0]:.2f}, {robot_coords[1]:.2f})', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.putText(frame, f'Confidence: {score:.2f}', (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Webcam', frame)

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
