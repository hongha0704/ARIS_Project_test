import numpy as np
import cv2 as cv
from ultralytics import YOLO

def pixel_to_mm(center_x_pixel, center_y_pixel):
    # center 좌표 변환(pixel to mm)
    center_x = -1 * ((center_x_pixel * (1 + (38 / 474) * (center_y_pixel / 194)) - 19 * (center_y_pixel / 194) - 62) * 800 / (474 * (1 + (38 / 474) * (center_y_pixel / 194)))) + 400
    center_y = ((center_y_pixel - 171) * (315 / 194)) - 170

    # 최종 center 좌표(mm)
    center_x_mm = center_x
    center_y_mm = center_y * 0.915

    return int(center_x_mm), int(center_y_mm)

def main():

    cap = cv.VideoCapture(2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # YOLO 모델 로드
    model = YOLO('/home/beakhongha/YOLO_ARIS/train18/weights/best.pt')

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
        results = model(frame)
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
                    center_x_mm, center_y_mm = pixel_to_mm(center_x_pixel, center_y_pixel)

                    # 결과 출력
                    print("카메라 좌표계 좌표:", [center_x_pixel, center_y_pixel])
                    print("로봇 좌표계 좌표:", [center_x_mm, center_y_mm])

                    # 화면에 바운딩 박스와 중심 좌표 표시
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.circle(frame, (center_x_pixel, center_y_pixel), 5, (0, 255, 0), -1)
                    cv.putText(frame, f'Capsule Coord: ({center_x_mm}, {center_y_mm})', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Webcam', frame)

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
