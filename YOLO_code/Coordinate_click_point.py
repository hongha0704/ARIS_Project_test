import cv2

# 클릭한 좌표를 저장하는 리스트
points = []

# 마우스 클릭 이벤트를 처리하는 함수
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 이벤트
        # 클릭한 좌표를 리스트에 추가
        points.append((x, y))

# 웹캠 장치 열기 (0은 기본 웹캠, 다른 장치를 사용할 경우 인덱스를 변경)
cap = cv2.VideoCapture(2)

# 웹캠이 열리지 않는 경우 에러 메시지 출력
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 창 생성 및 마우스 콜백 설정
cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', click_event)

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임을 읽지 못한 경우 루프 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 리스트에 저장된 모든 좌표에 점과 텍스트 표시
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.putText(frame, f'({point[0]}, {point[1]})', (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 프레임을 윈도우에 표시
    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
