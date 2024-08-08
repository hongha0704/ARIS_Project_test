from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("/home/beakhongha/YOLO_ARIS/train21/weights/best.pt")

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

# 글로벌 변수 설정
rois = []  # 여러 개의 ROI를 저장할 리스트
drawing = False  # ROI를 그리고 있는지 여부를 나타내는 플래그
ix, iy = -1, -1  # 마우스 클릭 시작 좌표

# 변수 초기화
A = False  # 첫 번째 ROI 내에서 객체가 인식되었는지 여부
B = False  # 두 번째 ROI 내에서 객체가 인식되었는지 여부
C = False  # 세 번째 ROI 내에서 객체가 인식되었는지 여부

# 마우스 콜백 함수 정의
def select_rois(event, x, y, flags, param):
    global ix, iy, drawing, rois

    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 왼쪽 버튼을 누르면 드로잉 시작
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 마우스가 움직일 때 사각형을 그림 (ROI 선택 영역)
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 0, 255), 1)
            cv2.imshow("Webcam", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # 마우스 왼쪽 버튼을 놓으면 드로잉 종료 및 ROI 추가
        drawing = False
        rois.append((ix, iy, x - ix, y - iy))
        print("Current ROIs:", rois)  # 현재 설정된 ROI를 출력

# 윈도우 생성 및 마우스 콜백 함수 설정
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", select_rois)

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = webcam.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
        break

    # 변수 초기화
    A = False
    B = False
    C = False

    # Run inference on the frame
    results = model(frame, imgsz=640, conf=0.5)  # 이미지 크기와 신뢰도(confidence)를 설정

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

            # ROI 내 객체 인식 확인
            for i, (rx, ry, rw, rh) in enumerate(rois[:3]):  # 최대 세 개의 ROI만 확인
                # 객체 박스와 ROI의 교차 영역 계산
                intersection_x1 = max(x1, rx)
                intersection_y1 = max(y1, ry)
                intersection_x2 = min(x2, rx + rw)
                intersection_y2 = min(y2, ry + rh)
                
                # 교차 영역의 면적 계산
                intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                
                # 객체 박스의 면적
                box_area = (x2 - x1) * (y2 - y1)
                
                # 교차 영역이 객체 박스 면적의 50% 이상일 때만 True로 설정
                if intersection_area >= 0.5 * box_area:
                    if i == 0:
                        A = True
                    elif i == 1:
                        B = True
                    elif i == 2:
                        C = True

    # 설정된 ROI를 빨간색 바운딩 박스로 그리기
    for (x, y, w, h) in rois:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 각 ROI를 빨간색 사각형으로 그림

    # 프레임 출력
    cv2.imshow("Webcam", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ROI 상태 및 현재 설정된 ROI 출력 (디버깅용)
    print(f"A: {A}, B: {B}, C: {C}")
    print("Current ROIs:", rois)

# 자원 해제
webcam.release()
cv2.destroyAllWindows()
