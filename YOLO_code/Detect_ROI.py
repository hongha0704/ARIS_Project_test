from ultralytics import YOLO
import cv2

global A_ZONE, B_ZONE, C_ZONE, NOT_SEAL

# Load a model
model = YOLO("/home/beakhongha/YOLO_ARIS/train1/weights/best.pt")

# Open the camera
webcam = cv2.VideoCapture(2)  # 0은 기본 카메라를 의미합니다. 만약 여러 개의 카메라가 연결되어 있다면, 1, 2 등을 시도해보세요.

# Set the width and height of the frames to be captured
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 너비를 640으로 설정
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 높이를 480으로 설정

# Check if the camera opened successfully
if not webcam.isOpened():
    print("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")
    exit()

# Define colors for different labels
colors = {
    'cup': (0, 0, 255),                 # 빨간색 (BGR 형식)
    'capsule': (255, 0, 0),             # 파란색 (BGR 형식)
    'capsule_label': (0, 255, 0),       # 녹색 (BGR 형식)
    'capsule_not_label': (0, 255, 255)  # 노란색 (BGR 형식)
}

# 영구적으로 설정된 ROI 구역
rois = [(455, 65, 95, 95), (360, 65, 95, 95), (265, 65, 95, 95)]  # A_ZONE, B_ZONE, C_ZONE 순서
seal_check_zone = (450, 230, 110, 110)  # Seal check ROI 구역
danger_zone1 = (40, 150, 525, 255)  # 추가된 구역
danger_zone2 = (85, 95, 200, 55)  # 추가된 구역

# 변수 초기화
A_ZONE = False  # 첫 번째 ROI 내에서 capsule 객체가 인식되었는지 여부
B_ZONE = False  # 두 번째 ROI 내에서 capsule 객체가 인식되었는지 여부
C_ZONE = False  # 세 번째 ROI 내에서 capsule 객체가 인식되었는지 여부
NOT_SEAL = False  # 특정 ROI 내에서 capsule_not_label 객체가 인식되었는지 여부

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = webcam.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
        break

    # Run inference on the frame
    results = model(frame, imgsz=640, conf=0.4)  # 이미지 크기와 신뢰도(confidence)를 설정

    # Display the results
    for result in results:
        boxes = result.boxes  # 인식된 객체의 바운딩 박스들
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 바운딩 박스 좌표
            conf = box.conf[0]  # 신뢰도
            cls = box.cls[0]  # 클래스 인덱스
            label = model.names[int(cls)]  # 클래스 라벨

            # Choose color based on label
            color = colors.get(label, (255, 255, 255))  # 라벨에 따른 색상 선택, 기본은 흰색

            # Draw the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 바운딩 박스 그리기
            cv2.putText(frame, f'{label}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 라벨 및 신뢰도 표시

            # ROI 내 객체 인식 확인 (capsule 객체만)
            if label == 'capsule':  # capsule 객체만 확인
                for i, (rx, ry, rw, rh) in enumerate(rois[:3]):  # 최대 세 개의 ROI만 확인
                    # ROI와 바운딩 박스의 교차 영역 계산
                    intersection_x1 = max(x1, rx)
                    intersection_y1 = max(y1, ry)
                    intersection_x2 = min(x2, rx + rw)
                    intersection_y2 = min(y2, ry + rh)

                    # 교차 영역의 면적 계산
                    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

                    # 바운딩 박스의 면적 계산
                    box_area = (x2 - x1) * (y2 - y1)

                    # 교차 영역이 바운딩 박스 면적의 50% 이상일 때만 True로 설정
                    if intersection_area >= 0.5 * box_area:
                        if i == 0 and not A_ZONE:
                            A_ZONE = True
                        elif i == 1 and not B_ZONE:
                            B_ZONE = True
                        elif i == 2 and not C_ZONE:
                            C_ZONE = True

            # 특정 ROI 내 capsule_not_label 객체 인식 확인
            if label == 'capsule_not_label':
                rx, ry, rw, rh = seal_check_zone
                # 특정 ROI와 바운딩 박스의 교차 영역 계산
                intersection_x1 = max(x1, rx)
                intersection_y1 = max(y1, ry)
                intersection_x2 = min(x2, rx + rw)
                intersection_y2 = min(y2, ry + rh)

                # 교차 영역의 면적 계산
                intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

                # 바운딩 박스의 면적 계산
                box_area = (x2 - x1) * (y2 - y1)

                # 교차 영역이 바운딩 박스 면적의 50% 이상일 때만 True로 설정
                if intersection_area >= 0.5 * box_area:
                    NOT_SEAL = True

    # 설정된 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
    for (x, y, w, h) in rois:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 각 ROI를 흰색 사각형으로 그림
    # seal_check_zone을 흰색 바운딩 박스로 그리고 선을 얇게 설정
    cv2.rectangle(frame, (seal_check_zone[0], seal_check_zone[1]), 
                    (seal_check_zone[0] + seal_check_zone[2], seal_check_zone[1] + seal_check_zone[3]), 
                    (255, 255, 255), 1)  # 특정 ROI를 흰색 사각형으로 그림
    # danger_zone1을 흰색 실선으로 그리기
    cv2.rectangle(frame, (danger_zone1[0], danger_zone1[1]), 
                    (danger_zone1[0] + danger_zone1[2], danger_zone1[1] + danger_zone1[3]), 
                    (255, 255, 255), 1)  # specific_roi2를 흰색 실선으로 그림
    # danger_zone2을 흰색 실선으로 그리기
    cv2.rectangle(frame, (danger_zone2[0], danger_zone2[1]), 
                    (danger_zone2[0] + danger_zone2[2], danger_zone2[1] + danger_zone2[3]), 
                    (255, 255, 255), 1)  # specific_roi3를 흰색 실선으로 그림

    # 프레임 출력
    cv2.imshow("Webcam", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ROI 상태 및 현재 설정된 ROI 출력 (디버깅용)
    print(f"A_ZONE: {A_ZONE}, B_ZONE: {B_ZONE}, C_ZONE: {C_ZONE}, NOT_SEAL: {NOT_SEAL}")
    # print("Current ROIs:", rois)

# 자원 해제
webcam.release()
cv2.destroyAllWindows()
