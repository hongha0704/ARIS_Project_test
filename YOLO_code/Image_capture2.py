import cv2
import time
import os

# 사진 저장 디렉토리 설정
output_dir = "Image_5"
os.makedirs(output_dir, exist_ok=True)

# 웹캠 초기화
webcam = cv2.VideoCapture(2)  # 기본 카메라 (2) 사용, 여러 카메라가 연결된 경우 인덱스를 조정

# 프레임 너비와 높이 설정
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 웹캠이 제대로 열렸는지 확인
if not webcam.isOpened():
    print("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")
    exit()

i = 0

# 프레임 읽기 루프
while True:
    ret, frame = webcam.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
        break

    # 프레임 표시
    cv2.imshow("Webcam", frame)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF

    # 'c' 키를 누르면 사진을 캡처하고 저장
    if key == ord('c'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"capture_{timestamp}.jpg")
        i += 1
        cv2.imwrite(file_path, frame)
        print(f"사진 저장: {file_path}, {i}")
        print(" ")

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

# 자원 해제
webcam.release()
cv2.destroyAllWindows()
